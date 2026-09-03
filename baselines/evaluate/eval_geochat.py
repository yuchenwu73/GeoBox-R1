#!/usr/bin/env python3
"""GeoChat-7B baseline: official weights (zero-shot) or the refGeo LoRA fine-tune.

Reproduces the "GeoChat" row (official `geochat-7B` weights) and, with
`--adapter_path`, the "GeoChat (SFT)" row: a LoRA trained on refGeo in GeoChat's own
grounding format (`baselines/finetune/prepare_geochat_data.py` + `run_geochat.sh`).
Both use the official referring prompt and output format, so they differ only in
how the weights are loaded.

Output format: `{<x1><y1><x2><y2>|<angle>}`, coordinates normalized to [0, 100] and
the angle in degrees. Boxes are mapped back with the ORIGINAL image size (the 504x504
model input is only a resize), which aligns them with the pixel ground truth.

Environment: the official GeoChat package (`geochat`), transformers 4.31.
Usage, from the repository root:
    python baselines/evaluate/eval_geochat.py --task hbb --model_path models/pretrained/geochat-7B --output_dir eval_results/baselines/geochat
    python baselines/evaluate/eval_geochat.py --task obb --model_path models/pretrained/geochat-7B \
        --adapter_path models/adapters/geochat_7b_refgeo_lora --output_dir eval_results/baselines/geochat_sft
"""
from __future__ import annotations

import argparse
import math
import re
from typing import List, Optional

from PIL import Image

from common import add_common_args, clip_hbb, run_evaluation, select_datasets


# --------------------------------------------------------------------------- parsing
def numbers(text: str) -> List[float]:
    return [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", text)]


def parse_nums(text: str) -> Optional[List[float]]:
    """Numbers of the first coordinate fragment: `{...}` (official), `<box>`, `[...]`, then the whole text."""
    for fragment in re.findall(r"\{[^{}]+\}|<box>.*?</box>|\[[^\[\]]+\]", text, flags=re.S | re.I) + [text]:
        values = numbers(fragment)
        if len(values) >= 4:
            return values
    return None


def rotate_rect(x1, y1, x2, y2, angle):
    """Four corners of the axis-aligned rectangle rotated by `angle` degrees about its center.

    Same convention as GeoChat/DOTA (`cv2.minAreaRect`): clockwise in image coordinates (y down).
    """
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    rad = math.radians(angle)
    c, s = math.cos(rad), math.sin(rad)
    return [[cx + (px - cx) * c - (py - cy) * s, cy + (px - cx) * s + (py - cy) * c]
            for px, py in ((x1, y1), (x2, y1), (x2, y2), (x1, y2))]


def hbb(text: str, size) -> Optional[List[float]]:
    """Parse an HBB in original-image pixels.

    GeoChat's output is really a rotated box. With a non-zero angle the horizontal
    box is the axis-aligned envelope of the rotated corners; using the two raw corner
    points instead under-estimates IoU systematically.
    """
    values = parse_nums(text)
    if not values:
        return None
    w, h = size
    x1, x2 = sorted((values[0] / 100 * w, values[2] / 100 * w))
    y1, y2 = sorted((values[1] / 100 * h, values[3] / 100 * h))
    if len(values) >= 5 and abs(values[4]) > 1e-3:
        pts = rotate_rect(x1, y1, x2, y2, values[4])
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    return clip_hbb([x1, y1, x2, y2], size)


def obb(text: str, size) -> Optional[List[List[float]]]:
    """Parse an OBB as four corners in original-image pixels.

    Strict protocol: only an output that carries the angle (5th value) counts; a box
    without an angle scores 0, so models without OBB capability are not credited.
    """
    values = parse_nums(text)
    if not values or len(values) < 5:
        return None
    w, h = size
    x1, x2 = sorted((values[0] / 100 * w, values[2] / 100 * w))
    y1, y2 = sorted((values[1] / 100 * h, values[3] / 100 * h))
    return rotate_rect(x1, y1, x2, y2, values[4])


def task_prompt(task: str, question: str) -> str:
    """The official GeoChat referring template; HBB and OBB share it and differ only in parsing."""
    return f"[refer] Give me the location of <p> {question} </p>"


# --------------------------------------------------------------------------- inference
class GeoChatEngine:
    """Official GeoChat weights; batching mirrors the official batch grounding script."""

    def __init__(self, model_path: str, conv_mode: str = "llava_v1"):
        import torch
        from geochat.model.builder import load_pretrained_model
        from geochat.utils import disable_torch_init

        disable_torch_init()
        # The vision tower path comes from `mm_vision_tower` in the checkpoint config.
        self.tok, self.model, self.processor, _ = load_pretrained_model(model_path, None, "geochat-7B")
        self.model.eval()
        self.conv_mode = conv_mode
        self.torch = torch

    def _build_ids(self, question: str):
        """Wrap the question with the image token and the llava_v1 template; return (input_ids, stop_str)."""
        from geochat.constants import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
        from geochat.conversation import SeparatorStyle, conv_templates
        from geochat.mm_utils import tokenizer_image_token

        if getattr(self.model.config, "mm_use_im_start_end", False):
            question = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + question
        else:
            question = DEFAULT_IMAGE_TOKEN + "\n" + question
        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        ids = tokenizer_image_token(conv.get_prompt(), self.tok, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
        stop = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        return ids, stop

    def infer_batch(self, paths: List[str], questions: List[str], max_tokens: int = 256) -> List[str]:
        torch = self.torch
        id_list, stop = [], None
        for question in questions:
            ids, stop = self._build_ids(question)
            id_list.append(ids)
        # Official batching: left-pad with id 0 to a common length, no attention mask.
        max_len = max(t.size(1) for t in id_list)
        padded = [torch.cat((torch.zeros((1, max_len - t.size(1)), dtype=t.dtype, device=t.device), t), dim=1)
                  for t in id_list]
        inputs = torch.cat(padded, dim=0)
        # Official preprocessing: 504x504 instead of the processor's 336 default.
        images = [Image.open(p).convert("RGB") for p in paths]
        pixels = self.processor.preprocess(images, crop_size={"height": 504, "width": 504},
                                           size={"shortest_edge": 504}, return_tensors="pt")["pixel_values"].half().cuda()
        with torch.inference_mode():
            out = self.model.generate(inputs, images=pixels, do_sample=False, num_beams=1,
                                      max_new_tokens=max_tokens, length_penalty=2.0, use_cache=True)
        # generate() returns the prompt as prefix; decode only the generated part.
        texts = self.tok.batch_decode(out[:, inputs.shape[1]:], skip_special_tokens=True)
        cleaned = []
        for text in texts:
            text = text.strip()
            if stop and text.endswith(stop):
                text = text[:-len(stop)].strip()
            cleaned.append(text)
        return cleaned


class GeoChatLoRAEngine(GeoChatEngine):
    """LoRA adapter merged onto the base weights through the official builder."""

    def __init__(self, adapter_path: str, base_path: str, conv_mode: str = "llava_v1"):
        import torch
        from geochat.model import GeoChatLlamaForCausalLM
        from geochat.model.builder import load_pretrained_model
        from geochat.utils import disable_torch_init

        disable_torch_init()
        # The released geochat-7B vision tower is 336-resolution (577x1024 position
        # embeddings) while CLIPVisionTower interpolates to 504 (1297x1024) at build
        # time, so strict loading fails on a size mismatch. Ignoring the mismatch keeps
        # the interpolated weights, matching the official inference path.
        # low_cpu_mem_usage/device_map must be off, otherwise the skipped parameters stay
        # on the meta device and the later .cuda() fails. The bicubic interpolation is
        # not implemented for fp16 on CPU, so load in fp32 and cast afterwards.
        orig = GeoChatLlamaForCausalLM.from_pretrained.__func__

        def patched(cls, *args, **kwargs):
            kwargs.setdefault("ignore_mismatched_sizes", True)
            kwargs["low_cpu_mem_usage"] = False
            kwargs.pop("device_map", None)
            kwargs["torch_dtype"] = torch.float32
            return orig(cls, *args, **kwargs)

        GeoChatLlamaForCausalLM.from_pretrained = classmethod(patched)
        # The model name must contain "geochat" (model class) and "lora" (merge branch).
        self.tok, self.model, self.processor, _ = load_pretrained_model(adapter_path, base_path, "geochat-7B-lora")
        self.model = self.model.half().cuda()
        self.model.eval()
        self.conv_mode = conv_mode
        self.torch = torch


# --------------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description="GeoChat-7B baseline evaluation")
    parser.add_argument("--task", choices=["hbb", "obb"], required=True)
    parser.add_argument("--model_path", required=True, help="Official geochat-7B weights (also the base for --adapter_path)")
    parser.add_argument("--adapter_path", default=None, help="refGeo LoRA directory (with non_lora_trainables.bin); omit for zero-shot")
    parser.add_argument("--conv_mode", default="llava_v1")
    add_common_args(parser, batch_size=8)
    args = parser.parse_args()

    if args.adapter_path:
        engine = GeoChatLoRAEngine(args.adapter_path, args.model_path, args.conv_mode)
        model_name = "GeoChat-7B (SFT)"
    else:
        engine = GeoChatEngine(args.model_path, args.conv_mode)
        model_name = "GeoChat-7B"
    parse = hbb if args.task == "hbb" else obb
    run_evaluation(
        args.task, select_datasets(args.task, args.dataset), engine.infer_batch,
        lambda question: task_prompt(args.task, question), parse, args, model_name,
        summary_extra={"adapter": args.adapter_path, "coordinate_mode": "norm100_original_image",
                       "prompt": "official [refer] <p>...</p>"},
    )


if __name__ == "__main__":
    main()
