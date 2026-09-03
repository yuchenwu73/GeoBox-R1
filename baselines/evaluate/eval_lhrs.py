#!/usr/bin/env python3
"""LHRS-Bot / LHRS-Bot-Nova baseline, HBB only.

Reproduces the "LHRS-Bot" and "LHRS-Bot-Nova" rows with the official checkpoints,
prompt templates and preprocessing (the model is built and prompted through the
official `lhrs` package, exactly as its own grounding evaluation does):

    lhrs       CLIP-L-224 tower,  prefix "[VG] ...",  output `[x1, y1, x2, y2]`
    lhrs-nova  SigLIP-384 tower,  prefix "[DET] ...", output `<bbox>[x1, y1, x2, y2]</bbox>`

Both emit coordinates normalized to [0, 1], mapped back with the original image size.
Neither model has an OBB output (their evaluation only knows axis-aligned boxes), so
OBB is not scored.

Environment: the official LHRS-Bot repository (`lhrs` package, ml_collections, yaml)
for the chosen model. Usage, from the repository root:
    python baselines/evaluate/eval_lhrs.py --model lhrs-nova --output_dir eval_results/baselines/lhrs_nova
"""
from __future__ import annotations

import argparse
import os
import re
from typing import List, Optional

from PIL import Image

from common import add_common_args, clip_hbb, run_evaluation, select_datasets

# Everything that differs between the two models. Paths are defaults under
# models/pretrained and can be overridden on the command line.
MODEL_SPECS = {
    "lhrs": dict(
        repo="models/pretrained/LHRS-Bot",
        ckpt="models/pretrained/LHRS-Bot-ckpt/FINAL.pt",
        text_path="models/pretrained/Llama-2-7b-chat-hf",
        vit_name="models/pretrained/clip-vit-large-patch14",
        prefix="[VG] Please output the coordinate of the following object: ",  # official DIOR eval template
        display="LHRS-Bot-7B",
    ),
    "lhrs-nova": dict(
        repo="models/pretrained/LHRS-Bot-Nova",
        ckpt="models/pretrained/LHRS-Bot-Nova-ckpt/Stage3/FINAL.pt",
        text_path="models/pretrained/Meta-Llama-3-8B-Instruct",
        vit_name="models/pretrained/siglip-so400m-patch14-384",
        prefix="[DET] Please output the coordinate of the following object: ",  # Nova renamed the task tag
        display="LHRS-Bot-Nova-8B",
    ),
}


# --------------------------------------------------------------------------- parsing
def parse_hbb(text: str, size, model_key: str) -> Optional[List[float]]:
    """Nova: prefer `<bbox>[...]</bbox>`; both: first `[...]` with 4+ numbers.

    Values are in [0, 1]; only the first four count (the official evaluation pops
    down to four as well). The box is scaled by the original image size and clipped.
    """
    w, h = size
    candidates = []
    if model_key == "lhrs-nova":
        candidates += re.findall(r"<bbox>\[(.*?)\]</bbox>", text, flags=re.S | re.I)
    candidates += re.findall(r"\[([0-9eE., \-]+)\]", text)
    for candidate in candidates:
        values = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?(?:[eE]-?\d+)?", candidate)]
        if len(values) < 4:
            continue
        return clip_hbb([values[0] * w, values[1] * h, values[2] * w, values[3] * h], size)
    return None


# --------------------------------------------------------------------------- inference
class LHRSEngine:
    """Model construction and prompt assembly reuse the official `lhrs` package (same distribution as main_vg.py)."""

    def __init__(self, model_key: str, spec: dict, config_path: str):
        import ml_collections
        import torch
        import yaml
        from lhrs.models import build_model
        from lhrs.utils import type_dict

        with open(config_path) as f:
            cfg = ml_collections.ConfigDict(yaml.safe_load(f))
        cfg.text.path = spec["text_path"]
        cfg.rgb_vision.vit_name = spec["vit_name"]
        cfg.stage = 0  # evaluation: LoRA is merged right after loading
        cfg.adjust_norm = False
        self.cfg = cfg
        self.torch = torch
        self.model_key = model_key

        self.model = build_model(cfg, activate_modal=("rgb", "text"))
        self.model.to(type_dict[cfg.dtype])
        msg = self.model.custom_load_state_dict(spec["ckpt"])
        if msg is not None:
            print(f"[load] missing={msg.missing_keys} unexpected={msg.unexpected_keys}")
        self.model.cuda().eval()

        # Vision preprocessing matches each model's tower: CLIP-224 or SigLIP-384.
        if model_key == "lhrs":
            from transformers import CLIPImageProcessor
            self.processor = CLIPImageProcessor.from_pretrained(spec["vit_name"])
        else:
            from transformers import SiglipImageProcessor
            self.processor = SiglipImageProcessor.from_pretrained(spec["vit_name"])
        self.tokenizer = self.model.text.tokenizer

        # Official prompt builders and collator (they pick the conversation template:
        # llava_llama_2 for LHRS-Bot, llama3 for Nova).
        from lhrs.Dataset.cap_dataset import DataCollatorForVGSupervisedDataset, preprocess, preprocess_multimodal
        self._preprocess = preprocess
        self._preprocess_mm = preprocess_multimodal
        self._collator = DataCollatorForVGSupervisedDataset(self.tokenizer)

    def _build_ids(self, question: str):
        """Same as VGEvalDataset.__getitem__: <image> first, then the official template."""
        conv = [dict(Question="<image>" + question, Answer=None)]
        conv = self._preprocess_mm(conv, tune_im_start=bool(self.cfg.tune_im_start))
        return self._preprocess(conv, self.tokenizer, has_image=True)["input_ids"][0]

    def infer_batch(self, paths: List[str], questions: List[str], max_tokens: int = 100) -> List[str]:
        torch = self.torch
        ids = [self._build_ids(q) for q in questions]
        images = [Image.open(p).convert("RGB") for p in paths]
        pixels = self.processor.preprocess(images, return_tensors="pt")["pixel_values"]
        # The collator takes (image, input_ids, target, filename) tuples and handles padding/masks.
        batch = self._collator([(pixels[i], ids[i], None, "") for i in range(len(ids))])
        image_t, input_ids, _, _, attention_mask = batch
        input_ids, attention_mask = input_ids.cuda(), attention_mask.cuda()
        image_t = image_t.to(dtype=next(self.model.parameters()).dtype).cuda()
        with torch.no_grad():
            out = self.model.generate(input_ids=input_ids, images=image_t, attention_mask=attention_mask,
                                      num_beams=1, do_sample=False, temperature=1.0, top_p=1.0,
                                      max_new_tokens=max_tokens)
        return [t.strip() for t in self.tokenizer.batch_decode(out, skip_special_tokens=True)]


# --------------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description="LHRS-Bot / LHRS-Bot-Nova baseline evaluation (HBB)")
    parser.add_argument("--model", choices=list(MODEL_SPECS), required=True)
    parser.add_argument("--repo", default=None, help="Official LHRS repository checkout (default: see MODEL_SPECS)")
    parser.add_argument("--ckpt", default=None, help="FINAL.pt checkpoint")
    parser.add_argument("--text_path", default=None, help="Language model directory")
    parser.add_argument("--vit_name", default=None, help="Vision tower directory")
    parser.add_argument("--config", default=None, help="multi_modal_eval.yaml; default: <repo>/Config/multi_modal_eval.yaml")
    add_common_args(parser, batch_size=8)
    args = parser.parse_args()

    spec = dict(MODEL_SPECS[args.model])
    for key in ("repo", "ckpt", "text_path", "vit_name"):
        if getattr(args, key):
            spec[key] = getattr(args, key)
    config_path = args.config or os.path.join(spec["repo"], "Config", "multi_modal_eval.yaml")
    engine = LHRSEngine(args.model, spec, config_path)

    run_evaluation(
        "hbb", select_datasets("hbb", args.dataset), engine.infer_batch,
        lambda question: spec["prefix"] + question,
        lambda text, size: parse_hbb(text, size, args.model),
        args, spec["display"],
        summary_extra={"coordinate_mode": "norm01_original_image", "prompt": spec["prefix"] + "<question>"},
    )


if __name__ == "__main__":
    main()
