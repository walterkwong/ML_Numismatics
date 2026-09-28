from __future__ import annotations

import logging

import torch
from peft import LoraConfig, get_peft_model
from torch import nn
from transformers import AutoImageProcessor, AutoModel, CLIPVisionModel


log = logging.getLogger(__name__)

BACKBONE_REGISTRY = {}


def register(name: str):
    def decorator(fn):
        BACKBONE_REGISTRY[name] = fn
        return fn

    return decorator


def _hf_processor_stats(
    processor,
    default_size: int = 224,
) -> tuple[int, list[float], list[float]]:
    size_dict = getattr(processor, "size", None) or {}
    size = (
        size_dict.get("shortest_edge")
        or size_dict.get("height")
        or default_size
    )

    mean = getattr(
        processor,
        "image_mean",
        [0.485, 0.456, 0.406],
    )
    std = getattr(
        processor,
        "image_std",
        [0.229, 0.224, 0.225],
    )

    return int(size), list(mean), list(std)


@register("dinov2")
def build_dinov2(pretrained: str = "facebook/dinov2-base"):
    model = AutoModel.from_pretrained(pretrained)
    processor = AutoImageProcessor.from_pretrained(pretrained)
    size, mean, std = _hf_processor_stats(processor)

    return model, model.config.hidden_size, size, mean, std


@register("ijepa")
def build_ijepa(pretrained: str = "facebook/ijepa_vith14_1k"):
    model = AutoModel.from_pretrained(pretrained)
    processor = AutoImageProcessor.from_pretrained(pretrained)
    size, mean, std = _hf_processor_stats(processor)

    return model, model.config.hidden_size, size, mean, std


@register("clip")
def build_clip(pretrained: str = "openai/clip-vit-base-patch32"):
    model = CLIPVisionModel.from_pretrained(pretrained)
    processor = AutoImageProcessor.from_pretrained(pretrained)
    size, mean, std = _hf_processor_stats(processor)

    return model, model.config.hidden_size, size, mean, std


@register("convnext_tiny")
def build_convnext(
    pretrained: str = "facebook/convnextv2-tiny-22k-224",
):
    model = AutoModel.from_pretrained(pretrained)
    processor = AutoImageProcessor.from_pretrained(pretrained)
    size, mean, std = _hf_processor_stats(processor)
    embed_dim = model.config.hidden_sizes[-1]

    return model, embed_dim, size, mean, std


def pooled_forward(
    backbone: nn.Module,
    pixel_values: torch.Tensor,
) -> torch.Tensor:
    outputs = backbone(pixel_values=pixel_values)

    if getattr(outputs, "pooler_output", None) is not None:
        return outputs.pooler_output

    hidden = outputs.last_hidden_state

    if hidden.dim() == 3:
        return hidden[:, 0]  # CLS token

    if hidden.dim() == 4:
        return hidden.mean(dim=(2, 3))  # GAP for ConvNeXt

    raise ValueError(f"Unexpected hidden state shape: {hidden.shape}")


LORA_TARGETS = ["q_proj", "v_proj", "query", "value"]


def apply_lora(
    backbone: nn.Module,
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
) -> nn.Module:
    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=LORA_TARGETS,
        bias="none",
    )

    log.info(f"Applying LoRA to: {LORA_TARGETS}")
    return get_peft_model(backbone, config)


def freeze(backbone: nn.Module) -> nn.Module:
    for param in backbone.parameters():
        param.requires_grad_(False)

    backbone.eval()
    return backbone


def build_backbone(name: str, adapt: str):
    backbone, embed_dim, size, mean, std = BACKBONE_REGISTRY[name]()

    if adapt == "linear":
        backbone = freeze(backbone)
    elif adapt == "lora":
        backbone = apply_lora(freeze(backbone))

    return backbone, embed_dim, size, mean, std