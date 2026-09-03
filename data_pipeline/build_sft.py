"""Step 3: assemble the SFT training sets (the paper's 2x2 data ablation).

Reads   <refgeo_root>/SFT/{RSVG,DIOR-RSVG}_HBB_train.jsonl            from build_hbb.py
        <refgeo_root>/SFT/{AVVG,VRSBench,GeoChat}_{OBB,CoT}_train.jsonl  from build_obb_cot.py
Writes  <output_dir>/sft/sft_<config>.jsonl

    config              order                      CoT supervision
    curriculum_cot      HBB -> OBB -> CoT, as is   yes     <- the main training set
    curriculum_no_cot   HBB -> OBB, as is          no (CoT samples become plain OBB samples)
    mixed_cot           globally shuffled          yes
    mixed_no_cot        globally shuffled          no

All four files contain the same 161,692 samples; only the ordering and the answer format
differ. The curriculum is the file order itself (easy horizontal boxes, then oriented boxes,
then the two-step refinement), so `sft.sh` trains with dataset and dataloader shuffling
disabled. `mixed_*` are the ablation arms.

Usage (from the repository root, after steps 1 and 2):
    python data_pipeline/build_sft.py                      # curriculum_cot only
    python data_pipeline/build_sft.py --config all
"""
import argparse
import copy
import json
import os
import random

from build_obb_cot import OBB_PROMPT, OBB_RESPONSE

# Prompt written into converted CoT records. The script version that built the released
# `*_no_cot` files had a stray blank line before the closing fence; it is kept verbatim so
# that `--config all` regenerates those files byte-for-byte (the ablation models were trained
# on them). Regular OBB records and the main `curriculum_cot` set use OBB_PROMPT unchanged.
CONVERTED_OBB_PROMPT = OBB_PROMPT.replace("]\n```", "]\n\n```")
assert CONVERTED_OBB_PROMPT != OBB_PROMPT

CONFIGS = {
    "curriculum_cot": {"cot": True, "shuffle": False},
    "curriculum_no_cot": {"cot": False, "shuffle": False},
    "mixed_cot": {"cot": True, "shuffle": True},
    "mixed_no_cot": {"cot": False, "shuffle": True},
}

# Concatenation order inside each block; it fixes the curriculum and the shuffle result.
HBB_FILES = ["RSVG_HBB_train.jsonl", "DIOR-RSVG_HBB_train.jsonl"]
OBB_FILES = ["AVVG_OBB_train.jsonl", "VRSBench_OBB_train.jsonl", "GeoChat_OBB_train.jsonl"]
COT_FILES = ["AVVG_CoT_train.jsonl", "VRSBench_CoT_train.jsonl", "GeoChat_CoT_train.jsonl"]


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(records, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(record, ensure_ascii=False) + "\n" for record in records)


def load_parts(sft_dir):
    parts = []
    for files in (HBB_FILES, OBB_FILES, COT_FILES):
        block = []
        for name in files:
            block.extend(load_jsonl(os.path.join(sft_dir, name)))
        parts.append(block)
    return parts  # [hbb, obb, cot]


def cot_to_obb(record):
    """Turn a CoT record into a plain OBB record: drop the horizontal box and the reasoning."""
    obb = copy.deepcopy(record)
    obb["messages"] = [
        {"role": "user", "content": CONVERTED_OBB_PROMPT},
        {"role": "assistant", "content": OBB_RESPONSE},
    ]
    obb["objects"]["bbox"] = record["objects"]["bbox"][1:5]  # [hbb, p1..p4] -> [p1..p4]
    return obb


def assemble(hbb, obb, cot, use_cot, shuffle, seed):
    if use_cot:
        data = hbb + obb + cot
    else:
        data = hbb + obb + [cot_to_obb(r) for r in cot]
    if shuffle:
        random.seed(seed)
        random.shuffle(data)
    return data


def main():
    parser = argparse.ArgumentParser(description="Assemble the SFT training sets.")
    parser.add_argument("--refgeo_root", default="data/refGeo", help="refGeo root; parts are read from <root>/SFT")
    parser.add_argument("--output_dir", default="data/GeoBox-R1-Data", help="sets are written to <output_dir>/sft")
    parser.add_argument("--config", default="curriculum_cot", choices=["all", *CONFIGS])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    hbb, obb, cot = load_parts(os.path.join(args.refgeo_root, "SFT"))
    total = len(hbb) + len(obb) + len(cot)
    print(f"Loaded {len(hbb)} HBB + {len(obb)} OBB + {len(cot)} CoT = {total} samples")

    names = list(CONFIGS) if args.config == "all" else [args.config]
    for name in names:
        cfg = CONFIGS[name]
        data = assemble(hbb, obb, cot, cfg["cot"], cfg["shuffle"], args.seed)
        path = os.path.join(args.output_dir, "sft", f"sft_{name}.jsonl")
        save_jsonl(data, path)
        order = "shuffled" if cfg["shuffle"] else "HBB -> OBB -> CoT" if cfg["cot"] else "HBB -> OBB"
        print(f"{name}: {len(data)} samples, {order} -> {path}")


if __name__ == "__main__":
    main()
