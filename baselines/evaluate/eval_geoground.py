#!/usr/bin/env python3
"""GeoGround baseline (LLaVA-v1.5-7B with the released task LoRA merged): HBB and OBB.

Reproduces the "GeoGround" rows with the official checkpoint and prompts:
    HBB  `[refer] output the bounding box of the <ref>...</ref> in the image.`
         -> `<box>[[x1, y1, x2, y2]]</box>`, normalized to [0, 1000]
    OBB  `[refer] output the oriented bounding box of the <ref>...</ref> in the image.`
         -> `<obb>[[cx, cy, w, h, angle]]</obb>`, "le90" box normalized to [0, 100]
Images are resized to 336x336 for the model, but the outputs are normalized, so they
are mapped back with the original image size to match the pixel ground truth.

Environment: the `llava` package that ships with GeoGround (transformers 4.57 works
through the forward patch below), plus opencv and shapely for OBB.
Usage, from the repository root:
    python baselines/evaluate/eval_geoground.py --task hbb --output_dir eval_results/baselines/geoground
    python baselines/evaluate/eval_geoground.py --task obb --output_dir eval_results/baselines/geoground
"""
from __future__ import annotations

import argparse
import functools
import inspect
import re
from typing import List, Optional, Sequence, Tuple

from PIL import Image

from common import add_common_args, clip_hbb, run_evaluation, select_datasets

DEFAULT_MODEL_PATH = "models/pretrained/llava-v1.5-7b-task-lora-geoground"

# Prompts copied from GeoGround's chat.py.
HBB_PROMPT = "[refer] output the bounding box of the <ref>{question}</ref> in the image."
OBB_PROMPT = "[refer] output the oriented bounding box of the <ref>{question}</ref> in the image."


def get_hbb_prompt(question: str) -> str:
    return HBB_PROMPT.format(question=question)


def get_obb_prompt(question: str) -> str:
    return OBB_PROMPT.format(question=question)


# --------------------------------------------------------------------------- parsing
def _numbers(text: str) -> List[float]:
    return [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", text)]


def _tag_content(text: str, tag_names: Sequence[str]) -> List[str]:
    out: List[str] = []
    for tag in tag_names:
        out.extend(re.findall(rf"<{tag}[^>]*>(.*?)</{tag}>", text, flags=re.I | re.S))
    return out


def _bracket_candidates(text: str) -> List[str]:
    cands: List[str] = []
    cands.extend(re.findall(r"\[\s*\[?\s*-?\d[\d\s,\.\-]*\]?\s*\]", text))
    cands.extend(re.findall(r"\(\s*-?\d[\d\s,\.\-]*\)", text))
    return cands


def _first_numeric_candidate(text: str, min_len: int, tag_names: Sequence[str]) -> Optional[List[float]]:
    """Numbers of the first fragment with enough values: tagged span, then brackets, then the whole text."""
    for candidate in _tag_content(text, tag_names) + _bracket_candidates(text) + [text]:
        values = _numbers(candidate)
        if len(values) >= min_len:
            return values
    return None


def parse_hbb_response(response: str, image_size: Tuple[int, int]) -> Optional[List[float]]:
    """`<box>[[x1, y1, x2, y2]]</box>` in [0, 1000] -> original-image pixels, clipped to the image."""
    values = _first_numeric_candidate(response, min_len=4, tag_names=["box", "hbb", "bbox"])
    if values is None:
        return None
    w, h = image_size
    box = [values[0] / 1000.0 * w, values[1] / 1000.0 * h, values[2] / 1000.0 * w, values[3] / 1000.0 * h]
    return [round(v, 2) for v in clip_hbb(box, image_size)]


def le90_to_polygon(cx: float, cy: float, bw: float, bh: float, angle_deg: float) -> List[List[float]]:
    """Corners of a (cx, cy, w, h, angle) rectangle through cv2.boxPoints.

    The le90 representation is ambiguous for horizontal or near-square targets (the
    same rectangle can be written with angle 0 or +-90 and swapped sides); boxPoints
    produces the right polygon for any angle, so nothing needs to be "fixed".
    """
    import cv2

    rect = ((float(cx), float(cy)), (max(1.0, float(bw)), max(1.0, float(bh))), float(angle_deg))
    return [[round(float(x), 2), round(float(y), 2)] for x, y in cv2.boxPoints(rect)]


def parse_obb_response(response: str, image_size: Tuple[int, int]) -> Optional[List[List[float]]]:
    """`<obb>[[cx, cy, w, h, angle]]</obb>` in [0, 100] -> four corners in original-image pixels.

    The center is scaled per axis. GeoGround only states that Text-OBB uses a
    resolution of 100 with a long-side-90 representation; w and h are edge lengths, so
    both are scaled by ONE length (the image width, which reproduces the published
    numbers). Scaling h by the image height instead squeezes the short side on
    non-square images such as AVVG (4000x2250) and lowers Acc@0.5 noticeably.
    Predicted polygons are not clipped: rotated IoU should see the raw geometry.
    An 8-number output is taken as four pixel corners.
    """
    values = _first_numeric_candidate(response, min_len=5, tag_names=["obb", "oriented_box", "box"])
    if values is None:
        return None
    w, h = image_size
    if len(values) >= 5 and len(values) != 8:
        cx, cy = values[0] / 100.0 * w, values[1] / 100.0 * h
        bw, bh = values[2] / 100.0 * w, values[3] / 100.0 * w
        return le90_to_polygon(cx, cy, bw, bh, values[4])
    if len(values) >= 8:
        return [[round(float(values[i]), 2), round(float(values[i + 1]), 2)] for i in range(0, 8, 2)]
    return None


# --------------------------------------------------------------------------- inference
def patch_llava_for_new_transformers() -> None:
    """Drop generate() kwargs (cache_position, logits_to_keep, ...) that the old LLaVA forward does not accept.

    The wrapper keeps the original signature because transformers validates model
    kwargs with inspect.signature.
    """
    from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM

    if getattr(LlavaLlamaForCausalLM.forward, "_geoground_patched", False):
        return
    orig_forward = LlavaLlamaForCausalLM.forward
    orig_sig = inspect.signature(orig_forward)
    allowed = set(orig_sig.parameters)

    @functools.wraps(orig_forward)
    def patched_forward(self, *args, **kwargs):
        kwargs = {k: v for k, v in kwargs.items() if k in allowed}
        return orig_forward(self, *args, **kwargs)

    patched_forward.__signature__ = orig_sig
    patched_forward._geoground_patched = True
    LlavaLlamaForCausalLM.forward = patched_forward


class GeoGroundModel:
    """GeoGround checkpoint with batched greedy decoding (extracted from chat.py)."""

    def __init__(self, model_path: str, conv_mode: str = "llava_v1"):
        import torch
        from llava.model.builder import load_pretrained_model
        from llava.utils import disable_torch_init

        patch_llava_for_new_transformers()
        self.torch = torch
        self.conv_mode = conv_mode
        disable_torch_init()
        # The checkpoint is fully merged: model_base=None and a model name without
        # "lora", otherwise the builder takes the non_lora_trainables/mm_projector branch.
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path, None, "llava-v1.5-7b-geoground")
        self.model.eval()
        # LLaVA re-pads the image-expanded embeddings following config.tokenizer_padding_side.
        # Decoder-only batches must be LEFT padded, or generation continues from pad
        # tokens and batch>1 results differ from batch=1; the default here is "right".
        self.model.config.tokenizer_padding_side = "left"
        self.tokenizer.padding_side = "left"
        self.pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

    def _build_prompt(self, question_with_refer: str) -> Tuple[str, str]:
        from llava.constants import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN
        from llava.conversation import SeparatorStyle, conv_templates

        if getattr(self.model.config, "mm_use_im_start_end", False):
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + question_with_refer
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + question_with_refer
        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        return conv.get_prompt(), stop_str

    def generate_batch(self, image_paths: List[str], prompts: List[str], max_new_tokens: int = 128) -> List[str]:
        """Greedy decoding for one batch; with left padding plus an attention mask, any batch size matches batch=1."""
        from llava.constants import IMAGE_TOKEN_INDEX
        from llava.mm_utils import tokenizer_image_token

        torch = self.torch
        built = [self._build_prompt(p) for p in prompts]
        stop_str = built[0][1]
        id_list = [tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                   for prompt, _ in built]
        max_len = max(t.size(0) for t in id_list)
        input_rows, attn_rows = [], []
        for t in id_list:
            n_pad = max_len - t.size(0)
            if n_pad > 0:
                pad = torch.full((n_pad,), self.pad_token_id, dtype=t.dtype)
                input_rows.append(torch.cat([pad, t], dim=0))
                attn_rows.append(torch.cat([torch.zeros(n_pad, dtype=torch.long), torch.ones(t.size(0), dtype=torch.long)], dim=0))
            else:
                input_rows.append(t)
                attn_rows.append(torch.ones(t.size(0), dtype=torch.long))
        input_ids = torch.stack(input_rows, dim=0).cuda()
        attention_mask = torch.stack(attn_rows, dim=0).cuda()

        # GeoGround inference uses a fixed 336x336 input before CLIP preprocessing.
        pils = [Image.open(p).convert("RGB").resize((336, 336)) for p in image_paths]
        image_tensor = self.image_processor.preprocess(
            pils, crop_size={"height": 336, "width": 336}, size={"shortest_edge": 336}, return_tensors="pt")["pixel_values"]

        with torch.inference_mode():
            output_ids = self.model.generate(
                inputs=input_ids, images=image_tensor.half().cuda(), attention_mask=attention_mask,
                do_sample=False, num_beams=1, max_new_tokens=max_new_tokens, use_cache=True)
        outs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        cleaned = []
        for out in outs:
            out = out.strip()
            if stop_str and out.endswith(stop_str):
                out = out[:-len(stop_str)].strip()
            if "ASSISTANT:" in out:  # some backends decode the role text as well
                out = out.split("ASSISTANT:")[-1].strip()
            cleaned.append(out)
        return cleaned


# --------------------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(description="GeoGround baseline evaluation")
    parser.add_argument("--task", choices=["hbb", "obb"], required=True)
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH, help="Merged GeoGround checkpoint directory")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Box outputs are short; 128 is enough")
    parser.add_argument("--conv_mode", default="llava_v1", help="LLaVA conversation template")
    parser.add_argument("--model_name", default="GeoGround", help="Model label in the result table")
    add_common_args(parser, batch_size=8)  # AVVG images are 4000x2250: batch 4 was used there
    args = parser.parse_args()

    engine = GeoGroundModel(args.model_path, conv_mode=args.conv_mode)

    def infer_batch(paths: List[str], prompts: List[str]) -> List[str]:
        return engine.generate_batch(paths, prompts, max_new_tokens=args.max_new_tokens)

    if args.task == "hbb":
        make_prompt, parse, coordinate_mode = get_hbb_prompt, parse_hbb_response, "norm1000_original_image"
    else:
        make_prompt, parse, coordinate_mode = get_obb_prompt, parse_obb_response, "norm100_le90_original_image"
    run_evaluation(
        args.task, select_datasets(args.task, args.dataset), infer_batch, make_prompt, parse, args, args.model_name,
        summary_extra={"model_path": args.model_path, "coordinate_mode": coordinate_mode},
    )


if __name__ == "__main__":
    main()
