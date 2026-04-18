from torchvision import models
import torch
model = models.efficientnet_v2_l(weights=models.EfficientNet_V2_L_Weights.IMAGENET1K_V1)
print("=== Backbone Structure ===")
for name, module in model.features.named_children():
    print(f"{name}: {module}")
    if hasattr(module, 'out_channels'):
        print(f"  Out channels: {module.out_channels}")
print("\n=== Testing forward pass ===")
x = torch.randn(1, 3, 320, 320)
with torch.no_grad():
    features = model.features(x)
print(f"Output shape: {features.shape}")
