"""Smoke check: encode a few training records with the ms-swift template and print them.

Shows exactly what the model sees after `<ref-object>` / `<bbox>` placeholders are expanded,
which is the quickest way to confirm a freshly built JSONL before launching training.

Usage (from the repository root):
    python data_pipeline/check_format.py
    python data_pipeline/check_format.py --dataset data/GeoBox-R1-Data/rl/rl_obb_20pct.jsonl --num_samples 3
"""
import argparse
import json
import os

# Must be set before ms-swift is imported: the image token budget matches the training
# launchers, and QWENVL_BBOX_FORMAT=new makes the Qwen-VL template render `<bbox>` as
# norm1000 coordinates in the grounding format used by GeoBox-R1.
os.environ.setdefault("IMAGE_MAX_TOKEN_NUM", "1024")
os.environ.setdefault("QWENVL_BBOX_FORMAT", "new")


def main():
    parser = argparse.ArgumentParser(description="Encode training records with the ms-swift template.")
    parser.add_argument("--model_path", default="models/pretrained/Qwen3-VL-4B-Instruct",
                        help="model directory; only its tokenizer/processor is loaded")
    parser.add_argument("--dataset", default="data/GeoBox-R1-Data/sft/sft_curriculum_cot.jsonl")
    parser.add_argument("--num_samples", type=int, default=2, help="records to encode from the start of the file")
    args = parser.parse_args()

    from swift import get_processor, get_template

    template = get_template(get_processor(args.model_path))
    template.set_mode("train")

    with open(args.dataset, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()][: args.num_samples]

    for index, record in enumerate(records):
        encoded = template.encode(record, return_template_inputs=True)
        print(f"===== sample {index} ({record.get('origin_dataset', '?')}) =====")
        print(f"[INPUT_IDS] {template.safe_decode(encoded['input_ids'])}\n")
        print(f"[LABELS] {template.safe_decode(encoded['labels'])}")
        print(f"[IMAGES] {encoded['template_inputs'].images}\n")


if __name__ == "__main__":
    main()
