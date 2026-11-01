"""ResNet-18 country classifier: ImageNet-initialised (fine-tuning) or randomly initialised (scratch).

The standard torchvision ResNet-18 is kept unchanged so the ImageNet weights load as they are; at
128 x 170 input the last feature map is 4 x 6 before global average pooling.
"""
import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

FEAT_DIM = 512


class CountryNet(nn.Module):
    def __init__(self, num_classes, init="imagenet", weights_path=None, dropout=0.2):
        super().__init__()
        if init == "imagenet":
            self.backbone = resnet18(weights=None)
            if weights_path:
                state = torch.load(weights_path, map_location="cpu", weights_only=True)
            else:
                state = ResNet18_Weights.IMAGENET1K_V1.get_state_dict(progress=True)
            self.backbone.load_state_dict(state)
        elif init == "scratch":
            # zero-init the last BN of each residual block: blocks start as identities
            self.backbone = resnet18(weights=None, zero_init_residual=True)
        else:
            raise ValueError(f"unknown init {init!r}")
        self.backbone.fc = nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(FEAT_DIM, num_classes)

    def forward(self, x, return_features=False):
        f = self.backbone(x)
        logits = self.head(self.dropout(f))
        return (logits, f) if return_features else logits


def param_groups(model, weight_decay, head_lr_mult):
    """Backbone / head groups; no weight decay on BatchNorm parameters and biases.
    Each group carries lr_mult, which the LR schedule multiplies in."""
    groups = {k: [] for k in ("bb_decay", "bb_no_decay", "head_decay", "head_no_decay")}
    for name, p in model.named_parameters():
        part = "head" if name.startswith("head.") else "bb"
        groups[f"{part}_{'no_decay' if p.ndim <= 1 else 'decay'}"].append(p)
    return [
        {"params": groups["bb_decay"], "weight_decay": weight_decay, "lr_mult": 1.0},
        {"params": groups["bb_no_decay"], "weight_decay": 0.0, "lr_mult": 1.0},
        {"params": groups["head_decay"], "weight_decay": weight_decay, "lr_mult": head_lr_mult},
        {"params": groups["head_no_decay"], "weight_decay": 0.0, "lr_mult": head_lr_mult},
    ]


STAGES = ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4"]


def freeze_backbone(model, k):
    """Freeze the stem and the first k residual stages (k=0: nothing, k=4: the whole backbone,
    i.e. a linear probe). Call after model.train(): frozen BatchNorms are put back in eval mode so
    their running statistics stay fixed."""
    frozen = set(STAGES[:k + 2]) if k > 0 else set()
    for name, mod in model.backbone.named_children():
        trainable = name not in frozen
        for p in mod.parameters():
            p.requires_grad_(trainable)
        mod.train(trainable and model.training)
