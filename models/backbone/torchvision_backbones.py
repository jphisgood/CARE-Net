"""Torchvision DeepLab/ResNet backbone with semantic and boundary features."""

import torch.nn as nn
import torchvision


def _build_deeplab_resnet(depth, use_coco_init):
    segmentation = torchvision.models.segmentation
    if depth == 101:
        constructor = segmentation.deeplabv3_resnet101
        weights_name = "DeepLabV3_ResNet101_Weights"
    elif depth == 50:
        constructor = segmentation.deeplabv3_resnet50
        weights_name = "DeepLabV3_ResNet50_Weights"
    else:
        raise ValueError("Unsupported ResNet depth: %s" % depth)

    weights_enum = getattr(segmentation, weights_name, None)
    if weights_enum is not None:
        weights = weights_enum.DEFAULT if use_coco_init else None
        return constructor(
            weights=weights,
            weights_backbone=None,
            progress=True,
            num_classes=21,
            aux_loss=True,
        )

    return constructor(
        pretrained=use_coco_init,
        pretrained_backbone=False,
        progress=True,
        num_classes=21,
        aux_loss=True,
    )


class _TVDeepLabEncoder(nn.Module):
    def __init__(self, depth, use_coco_init, low_dim=64):
        super().__init__()
        model = _build_deeplab_resnet(depth, use_coco_init)
        if use_coco_init:
            print("###### NETWORK: Using MS-COCO initialization ######")
        else:
            print("###### NETWORK: Training from scratch ######")

        # Keep the old attribute names so V3 checkpoints can warm-start V4.
        self.backbone = model.backbone
        self.localconv = nn.Conv2d(2048, 256, kernel_size=1, bias=False)
        self.lowconv = nn.Sequential(
            nn.Conv2d(256, low_dim, kernel_size=1, bias=False),
            nn.GroupNorm(8, low_dim),
            nn.ReLU(inplace=True),
        )

    def _manual_forward(self, x):
        # IntermediateLayerGetter stores the original ResNet children under these
        # names.  Running them explicitly exposes layer1 without relying on the
        # optional torchvision auxiliary output (which is layer3, not low-level).
        modules = self.backbone
        x = modules["conv1"](x)
        x = modules["bn1"](x)
        x = modules["relu"](x)
        x = modules["maxpool"](x)
        low_raw = modules["layer1"](x)
        x = modules["layer2"](low_raw)
        x = modules["layer3"](x)
        high_raw = modules["layer4"](x)
        return high_raw, low_raw

    def forward(self, x_in, low_level=False):
        high_raw, low_raw = self._manual_forward(x_in)
        high = self.localconv(high_raw)
        if not low_level:
            return high
        low = self.lowconv(low_raw)
        return high, low


class TVDeeplabRes101Encoder(_TVDeepLabEncoder):
    def __init__(self, use_coco_init, aux_dim_keep=64):
        super().__init__(101, use_coco_init, low_dim=aux_dim_keep)


class TVDeeplabRes50Encoder(_TVDeepLabEncoder):
    def __init__(self, use_coco_init, aux_dim_keep=64):
        super().__init__(50, use_coco_init, low_dim=aux_dim_keep)
