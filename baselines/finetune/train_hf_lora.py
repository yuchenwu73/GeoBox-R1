#!/usr/bin/env python3
"""LoRA SFT for Hugging Face VLM checkpoints (InternVL3-hf, LLaVA-OneVision-1.5, ...).

Used for the "InternVL3 (SFT)" and "LLaVA-OV-1.5 (SFT)" baseline rows; see run_internvl3.sh
and run_llava_ov15.sh for the exact settings.

Data: JSONL produced by prepare_hf_data.py, one record per line:
    {"id", "image": "<Subset>/<file>" (relative to --image-root),
     "conversations": [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}]}
The user turn carries a single image (a leading "<image>" marker is stripped and the image is
passed through the processor's chat template).

Trainable weights: LoRA on every linear layer of the language model only (LLM_TARGET_PATTERN);
the vision tower and the projector stay frozen. Labels are masked to the assistant turn: the
prompt prefix is tokenized separately and its length is set to -100, as is padding.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from PIL import Image
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    set_seed,
)

LLM_TARGET_PATTERN = (
    r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-steps", type=int, default=-1, help="Positive value overrides epochs; for smoke tests")
    parser.add_argument("--crop-to-patches", action="store_true", help="Enable dynamic high-resolution tiling (InternVL multi-patch input; the patch budget comes from the processor's max_patches)")
    parser.add_argument("--lora-target-regex", default=None, help="Override the LoRA target-module regex (default: every linear layer of the language model)")
    parser.add_argument("--max-pixels", type=int, default=None,
                        help="Override max_pixels of Qwen2-VL-style image processors to cap the image token budget (tokens = pixels / 784)")
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--bits", type=int, choices=(4, 8, 16), default=16)
    parser.add_argument("--attn-implementation", default="sdpa", choices=("eager", "sdpa", "flash_attention_2"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-to", default="none")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def normalize_messages(conversations: list[dict[str, str]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for message in conversations:
        role = message["role"]
        text = message["content"]
        if role == "user":
            text = text.replace("<image>", "", 1).lstrip("\n ")
            content: Any = [{"type": "image"}, {"type": "text", "text": text}]
        else:
            content = [{"type": "text", "text": text}]
        result.append({"role": role, "content": content})
    return result


class MultimodalSFTCollator:
    def __init__(self, processor: Any, image_root: Path, max_length: int, crop_to_patches: bool = False) -> None:
        self.processor = processor
        self.image_root = image_root
        self.max_length = max_length
        self.crop_to_patches = crop_to_patches

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        images = []
        full_texts = []
        prompt_texts = []
        for example in examples:
            image_path = self.image_root / example["image"]
            with Image.open(image_path) as image:
                images.append(image.convert("RGB").copy())
            messages = normalize_messages(example["conversations"])
            if messages[-1]["role"] != "assistant":
                raise ValueError(f"last message must be assistant: {example.get('id')}")
            full_texts.append(
                self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            )
            prompt_texts.append(
                self.processor.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
            )

        # Only InternVL processors accept crop_to_patches; fixed-resolution models (LLaVA) must not receive it.
        image_kwargs = {"crop_to_patches": True} if self.crop_to_patches else {}
        batch = self.processor(
            text=full_texts,
            images=images,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            **image_kwargs,
        )
        prompt_batch = self.processor(
            text=prompt_texts,
            images=images,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            **image_kwargs,
        )
        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100
        prompt_lengths = prompt_batch["attention_mask"].sum(dim=1).tolist()
        for row, prompt_length in enumerate(prompt_lengths):
            labels[row, : int(prompt_length)] = -100
        batch["labels"] = labels
        return batch


def validate_dataset(dataset_path: Path, image_root: Path, limit: int = 16) -> None:
    checked = 0
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not (image_root / record["image"]).is_file():
                raise FileNotFoundError(image_root / record["image"])
            conversations = record.get("conversations", [])
            if len(conversations) < 2 or conversations[0].get("role") != "user" or conversations[-1].get("role") != "assistant":
                raise ValueError(f"invalid conversation at line {line_number}")
            checked += 1
            if checked >= limit:
                break
    if checked == 0:
        raise ValueError(f"empty dataset: {dataset_path}")
    print(f"validated {checked} records from {dataset_path}")


def main() -> None:
    args = parse_args()
    dataset_path = args.dataset.expanduser().resolve()
    image_root = args.image_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    validate_dataset(dataset_path, image_root)
    if args.validate_only:
        return

    set_seed(args.seed)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "right"
        processor.tokenizer.model_max_length = args.max_length
    if args.max_pixels is not None and hasattr(processor, "image_processor") \
            and hasattr(processor.image_processor, "max_pixels"):
        processor.image_processor.max_pixels = args.max_pixels
        if isinstance(getattr(processor.image_processor, "size", None), dict):
            processor.image_processor.size["longest_edge"] = args.max_pixels

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    quantization_config = None
    if args.bits in (4, 8):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=args.bits == 4,
            load_in_8bit=args.bits == 8,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )

    try:
        model = AutoModelForImageTextToText.from_pretrained(
            args.model,
            torch_dtype=dtype,
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            attn_implementation=args.attn_implementation,
        )
    except (ValueError, KeyError):
        # Custom architectures such as LLaVA-OneVision-1.5 only register AutoModelForCausalLM in auto_map.
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=dtype,
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            attn_implementation=args.attn_implementation,
        )
    model.config.use_cache = False
    if args.bits in (4, 8):
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            model.get_input_embeddings().register_forward_hook(
                lambda _module, _inputs, output: output.requires_grad_(True)
            )
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_regex or LLM_TARGET_PATTERN,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = load_dataset("json", data_files=str(dataset_path), split="train")
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=False,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        weight_decay=0.0,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=dtype == torch.bfloat16,
        fp16=dtype == torch.float16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        dataloader_num_workers=args.workers,
        report_to=args.report_to,
        ddp_find_unused_parameters=False,
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=MultimodalSFTCollator(processor, image_root, args.max_length, args.crop_to_patches),
    )
    trainer.train(resume_from_checkpoint=True if list(output_dir.glob("checkpoint-*")) else None)
    trainer.save_state()
    trainer.save_model()
    processor.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
