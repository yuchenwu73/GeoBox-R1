#!/usr/bin/env python3
"""Thin entry point for GeoChat LoRA training; the official geochat.train.train() does the work.

The released geochat-7B checkpoint stores a 336-resolution vision tower (position embedding
577x1024), while GeoChat's CLIPVisionTower is built at 504 (1297x1024), so a strict
from_pretrained() fails with a size mismatch. Injecting ignore_mismatched_sizes=True keeps the
interpolated 504 embedding that the tower creates at construction time; every other vision
weight matches CLIP and loads normally. train() then calls initialize_vision_modules(), which
reloads the tower from --vision_tower and interpolates it again, exactly as the official
inference path does.
"""

from geochat.model import GeoChatLlamaForCausalLM
from geochat.train.train import train

_orig_from_pretrained = GeoChatLlamaForCausalLM.from_pretrained


def _patched_from_pretrained(*args, **kwargs):
    kwargs.setdefault("ignore_mismatched_sizes", True)
    return _orig_from_pretrained(*args, **kwargs)


GeoChatLlamaForCausalLM.from_pretrained = _patched_from_pretrained

if __name__ == "__main__":
    train()
