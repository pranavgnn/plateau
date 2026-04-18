"""PyTorch model for license plate detection with FPN + CBAM + CIoU."""

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
    """Compute Complete IoU (CIoU) - better for bbox regression than IoU/GIoU."""
    # IoU
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

    # Enclosing box
    enc_xmin = torch.min(y_true[:, 0], y_pred[:, 0])
    enc_ymin = torch.min(y_true[:, 1], y_pred[:, 1])
    enc_xmax = torch.max(y_true[:, 2], y_pred[:, 2])
    enc_ymax = torch.max(y_true[:, 3], y_pred[:, 3])
    enc_w = torch.clamp(enc_xmax - enc_xmin, min=0.0)
    enc_h = torch.clamp(enc_ymax - enc_ymin, min=0.0)
    enc_diag = enc_w ** 2 + enc_h ** 2

    # Center distance
    true_cx = (y_true[:, 0] + y_true[:, 2]) / 2.0
    true_cy = (y_true[:, 1] + y_true[:, 3]) / 2.0
    pred_cx = (y_pred[:, 0] + y_pred[:, 2]) / 2.0
    pred_cy = (y_pred[:, 1] + y_pred[:, 3]) / 2.0
    center_dist = (true_cx - pred_cx) ** 2 + (true_cy - pred_cy) ** 2

    # Aspect ratio
    v = (4.0 / (np.pi ** 2)) * (
        torch.atan(true_w / torch.clamp(true_h, min=1e-6)) - 
        torch.atan(pred_w / torch.clamp(pred_h, min=1e-6))
    ) ** 2
    alpha = torch.divide(v, (1.0 - iou + v) + 1e-6)

    ciou = iou - (center_dist / (enc_diag + 1e-6)) - alpha * v
    return torch.clamp(ciou, min=-1.0, max=1.0)


class ChannelAttention(nn.Module):
    """CBAM Channel Attention Module."""
    
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
        
        avg_out = self.avg_pool(x).view(batch, channels)
        avg_out = self.fc(avg_out).view(batch, channels, 1, 1)
        
        max_out = self.max_pool(x).view(batch, channels)
        max_out = self.fc(max_out).view(batch, channels, 1, 1)
        
        out = self.sigmoid(avg_out + max_out)
        return x * out


class SpatialAttention(nn.Module):
    """CBAM Spatial Attention Module."""
    
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        spatial_out = self.sigmoid(self.conv(x_cat))
        return x * spatial_out


class CBAM(nn.Module):
    """Convolutional Block Attention Module (Channel + Spatial)."""
    
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.channel_att = ChannelAttention(channels, reduction=reduction)
        self.spatial_att = SpatialAttention()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x


class BoxRegressionLoss(nn.Module):
    """Complete IoU (CIoU) loss - better for bbox regression."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        ciou = box_ciou(y_true, y_pred)
        return torch.mean(1.0 - ciou)


def create_optimizer(model: nn.Module, learning_rate: float, weight_decay: float = 1e-5) -> torch.optim.Optimizer:
    """Create AdamW optimizer."""
    return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


class LicensePlateDetectionModel(nn.Module):
    """EfficientNetV2 + FPN + CBAM + Dense head for plate localization."""

    def __init__(self, input_shape: Tuple[int, int, int] = (320, 320, 3), device: torch.device | None = None) -> None:
        super().__init__()
        self.input_shape = input_shape
        self.device = device or (torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.backbone: Optional[nn.Module] = None
        
        # Build layers
        self._build_backbone()
        self._build_fpn()
        self._build_head()
        
        self.to(self.device)

    def _build_backbone(self) -> None:
        """Build EfficientNetV2L backbone."""
        # Load pretrained EfficientNetV2L
        efficientnet = models.efficientnet_v2_l(weights=models.EfficientNet_V2_L_Weights.IMAGENET1K_V1)
        
        # Remove classification head
        self.backbone = nn.Sequential(*list(efficientnet.children())[:-1])

    def _build_fpn(self) -> None:
        """Build FPN head."""
        # efficientnet_v2_l outputs 1280 channels at final layer
        self.fpn_conv = nn.Conv2d(1280, 256, kernel_size=1, padding=0)
        self.fpn_refine = nn.Conv2d(256, 256, kernel_size=3, padding=1)

    def _build_head(self) -> None:
        """Build detection head."""
        # CBAM on 256 channels
        self.cbam = CBAM(256, reduction=16)
        
        # Dense head - input is 256 (from global avg pool)
        self.head = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.SiLU(inplace=True),
            nn.Dropout(0.4),
            
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.SiLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(64, 4),
            nn.Sigmoid(),  # Output normalized [0, 1] for bbox
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Get backbone features [B, 1280, 10, 10]
        features = self.backbone[0](x)
        
        # FPN: 1x1 conv to reduce channels
        feat = self.fpn_conv(features)
        feat = self.fpn_refine(feat)
        
        # Global average pooling [B, 256, 1, 1] -> [B, 256]
        feat = F.adaptive_avg_pool2d(feat, (1, 1))
        feat = feat.view(feat.size(0), -1)
        
        # CBAM (reshape to spatial, apply, reshape back)
        feat_spatial = feat.view(feat.size(0), 256, 1, 1)
        feat_spatial = self.cbam(feat_spatial)
        feat = feat_spatial.view(feat_spatial.size(0), -1)
        
        # Dense head for bbox regression
        bbox = self.head(feat)
        return bbox

    def freeze_backbone(self, freeze: bool = True) -> None:
        """Freeze or unfreeze backbone."""
        for param in self.backbone.parameters():
            param.requires_grad = not freeze

    def freeze_top_layers(self, fine_tune_at: int) -> None:
        """Freeze layers up to fine_tune_at."""
        layer_count = 0
        for param in self.backbone.parameters():
            if layer_count < fine_tune_at:
                param.requires_grad = False
                layer_count += 1
            else:
                param.requires_grad = True

    def save(self, filepath: str) -> None:
        """Save model to disk."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'model_config': {
                'input_shape': self.input_shape,
            }
        }, filepath)
        print(f"Model saved to {filepath}")

    def load(self, filepath: str) -> None:
        """Load model from disk."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.to(self.device)
        print(f"Model loaded from {filepath}")

    def get_summary(self) -> None:
        """Print model summary."""
        print(f"Model: LicensePlateDetectionModel")
        print(f"Device: {self.device}")
        print(f"Input shape: {self.input_shape}")
        print(f"Total parameters: {sum(p.numel() for p in self.parameters()):,}")
        print(f"Trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")
