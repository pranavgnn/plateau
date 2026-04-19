"""Improved PyTorch model for license plate detection with confidence scoring."""

from __future__ import annotations

from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def box_iou(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """Compute IoU for normalized corner-format boxes."""
    inter_xmin = torch.max(y_true[:, 0], y_pred[:, 0])
    inter_ymin = torch.max(y_true[:, 1], y_pred[:, 1])
    inter_xmax = torch.min(y_true[:, 2], y_pred[:, 2])
    inter_ymax = torch.min(y_true[:, 3], y_pred[:, 3])

    inter_w = torch.clamp(inter_xmax - inter_xmin, min=0.0)
    inter_h = torch.clamp(inter_ymax - inter_ymin, min=0.0)
    intersection = inter_w * inter_h

    true_w = torch.clamp(y_true[:, 2] - y_true[:, 0], min=0.0)
    true_h = torch.clamp(y_true[:, 3] - y_true[:, 1], min=0.0)
    pred_w = torch.clamp(y_pred[:, 2] - y_pred[:, 0], min=0.0)
    pred_h = torch.clamp(y_pred[:, 3] - y_pred[:, 1], min=0.0)

    true_area = true_w * true_h
    pred_area = pred_w * pred_h
    union = true_area + pred_area - intersection
    return torch.divide(intersection, union + 1e-6)


def box_ciou(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """Compute Complete IoU (CIoU) - better for bbox regression."""
    inter_xmin = torch.max(y_true[:, 0], y_pred[:, 0])
    inter_ymin = torch.max(y_true[:, 1], y_pred[:, 1])
    inter_xmax = torch.min(y_true[:, 2], y_pred[:, 2])
    inter_ymax = torch.min(y_true[:, 3], y_pred[:, 3])

    inter_w = torch.clamp(inter_xmax - inter_xmin, min=0.0)
    inter_h = torch.clamp(inter_ymax - inter_ymin, min=0.0)
    intersection = inter_w * inter_h

    true_w = torch.clamp(y_true[:, 2] - y_true[:, 0], min=1e-6)
    true_h = torch.clamp(y_true[:, 3] - y_true[:, 1], min=1e-6)
    pred_w = torch.clamp(y_pred[:, 2] - y_pred[:, 0], min=1e-6)
    pred_h = torch.clamp(y_pred[:, 3] - y_pred[:, 1], min=1e-6)

    true_area = true_w * true_h
    pred_area = pred_w * pred_h
    union = true_area + pred_area - intersection
    iou = torch.divide(intersection, union + 1e-6)

    enc_xmin = torch.min(y_true[:, 0], y_pred[:, 0])
    enc_ymin = torch.min(y_true[:, 1], y_pred[:, 1])
    enc_xmax = torch.max(y_true[:, 2], y_pred[:, 2])
    enc_ymax = torch.max(y_true[:, 3], y_pred[:, 3])
    enc_w = torch.clamp(enc_xmax - enc_xmin, min=0.0)
    enc_h = torch.clamp(enc_ymax - enc_ymin, min=0.0)
    enc_diag = enc_w ** 2 + enc_h ** 2

    true_cx = (y_true[:, 0] + y_true[:, 2]) / 2.0
    true_cy = (y_true[:, 1] + y_true[:, 3]) / 2.0
    pred_cx = (y_pred[:, 0] + y_pred[:, 2]) / 2.0
    pred_cy = (y_pred[:, 1] + y_pred[:, 3]) / 2.0
    center_dist = (true_cx - pred_cx) ** 2 + (true_cy - pred_cy) ** 2

    v = (4.0 / (np.pi ** 2)) * (
        torch.atan(true_w / torch.clamp(true_h, min=1e-6)) - 
        torch.atan(pred_w / torch.clamp(pred_h, min=1e-6))
    ) ** 2
    alpha = torch.divide(v, (1.0 - iou + v) + 1e-6)

    ciou = iou - (center_dist / (enc_diag + 1e-6)) - alpha * v
    return torch.clamp(ciou, min=-1.0, max=1.0)


class ChannelAttention(nn.Module):
    """CBAM Channel Attention."""
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(batch, channels))
        max_out = self.fc(self.max_pool(x).view(batch, channels))
        out = avg_out + max_out
        return x * self.sigmoid(out).view(batch, channels, 1, 1)


class SpatialAttention(nn.Module):
    """CBAM Spatial Attention."""
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(x_cat)
        return x * self.sigmoid(out)


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""
    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7) -> None:
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class LicensePlateDetectionModelImproved(nn.Module):
    """Improved license plate detection with confidence scoring and better architecture."""
    
    def __init__(self, input_shape: Tuple[int, int, int] = (320, 320, 3), device: torch.device | None = None) -> None:
        super().__init__()
        self.input_shape = input_shape
        self.device = device or (torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        
        # Build layers
        self._build_backbone()
        self._build_fpn()
        self._build_heads()
        self.to(self.device)

    def _build_backbone(self) -> None:
        """Build EfficientNetV2M backbone (better than L for small objects)."""
        efficientnet = models.efficientnet_v2_m(weights=models.EfficientNet_V2_M_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(efficientnet.children())[:-1])

    def _build_fpn(self) -> None:
        """Build FPN with multi-scale features."""
        # EfficientNetV2M outputs 1280 channels
        self.fpn_conv = nn.Conv2d(1280, 512, kernel_size=1, padding=0)
        self.fpn_refine = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
        )
        self.cbam = CBAM(256, reduction=16)

    def _build_heads(self) -> None:
        """Build confidence and bbox heads."""
        # Shared feature processing
        self.feature_process = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(inplace=True),
            nn.Dropout(0.3),
        )
        
        # Confidence head (objectness)
        self.confidence_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.SiLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid(),  # Output [0, 1] confidence
        )
        
        # BBox regression head
        self.bbox_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.SiLU(inplace=True),
            nn.Linear(128, 4),
            nn.Sigmoid(),  # Output normalized [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Returns [batch, 5] with [x1, y1, x2, y2, confidence]."""
        # Backbone
        features = self.backbone[0](x)
        
        # FPN
        feat = self.fpn_conv(features)
        feat = self.fpn_refine(feat)
        feat = self.cbam(feat)
        
        # Process features
        processed = self.feature_process(feat)
        
        # Get predictions
        confidence = self.confidence_head(processed)
        bbox = self.bbox_head(processed)
        
        # Combine: [x1, y1, x2, y2, confidence]
        output = torch.cat([bbox, confidence], dim=1)
        return output

    def save(self, filepath: str) -> None:
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'input_shape': self.input_shape,
        }, filepath)
        print(f"Model saved to {filepath}")

    def load(self, filepath: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {filepath}")

    def freeze_backbone(self, freeze: bool = True) -> None:
        """Freeze/unfreeze backbone."""
        for param in self.backbone.parameters():
            param.requires_grad = not freeze

    def freeze_top_layers(self, fine_tune_at: int = 200) -> None:
        """Freeze layers up to fine_tune_at."""
        layer_count = 0
        for layer in self.backbone.modules():
            if isinstance(layer, (nn.Conv2d, nn.BatchNorm2d)):
                layer_count += 1
                if layer_count < fine_tune_at:
                    for param in layer.parameters():
                        param.requires_grad = False

    def get_summary(self) -> None:
        """Print model summary."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")


class ImprovedBoxRegressionLoss(nn.Module):
    """Loss combining CIoU bbox loss and confidence loss."""
    
    def __init__(self, bbox_weight: float = 1.0, conf_weight: float = 0.5) -> None:
        super().__init__()
        self.bbox_weight = bbox_weight
        self.conf_weight = conf_weight

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_true: [batch, 4] ground truth boxes
            y_pred: [batch, 5] predictions [x1, y1, x2, y2, confidence]
        """
        # Extract predictions
        bbox_pred = y_pred[:, :4]
        conf_pred = y_pred[:, 4]
        
        # CIoU loss for bboxes
        ciou = box_ciou(y_true, bbox_pred)
        bbox_loss = 1.0 - ciou.mean()
        
        # Binary cross entropy loss for confidence (all ground truth plates = 1.0)
        conf_true = torch.ones_like(conf_pred)
        conf_loss = F.binary_cross_entropy(conf_pred, conf_true)
        
        # Combined loss
        total_loss = self.bbox_weight * bbox_loss + self.conf_weight * conf_loss
        return total_loss


def create_optimizer(model: nn.Module, learning_rate: float = 1e-3) -> torch.optim.Optimizer:
    """Create optimizer with weight decay."""
    return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
