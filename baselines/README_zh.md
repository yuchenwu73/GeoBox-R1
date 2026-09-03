<a href="README.md">English</a> | <b>简体中文</b>

# 基线

论文中用于对比的模型的评测与微调代码。

## evaluate/ — 基线评测

与 `evaluation/` 中的主评测脚本口径完全一致：一次跑出 7 个 HBB、3 个 OBB 评测集上的
Acc@0.5、Acc@0.7 与 mIoU / mRIoU。`common.py` 放公共流程（读取 metainfo、批推理与逐样本重试、
IoU / 旋转 IoU、结果落盘），每个脚本只封装一种模型的推理接口、提示词与坐标制式。

| 脚本 | 对应论文行 | 解析的模型输出 |
| --- | --- | --- |
| `eval_geochat.py` | GeoChat（官方权重）；加 `--adapter_path` 为 GeoChat (SFT) | `{<x1><y1><x2><y2>\|<angle>}`，0–100 网格 |
| `eval_geoground.py --task hbb\|obb` | GeoGround | HBB `<box>[[x1,y1,x2,y2]]</box>`（norm1000）；OBB `<obb>[[cx,cy,w,h,θ]]</obb>`（norm100） |
| `eval_internvl.py` | InternVL3-2B / 8B 零样本 | `<box>[[x1,y1,x2,y2]]</box>`（norm1000） |
| `eval_hf.py` | LLaVA-OV-1.5-4B / 8B 零样本；InternVL3-2B / 8B (SFT)；LLaVA-OV-1.5-4B (SFT) | 主模型的 JSON 格式，坐标空间由 `--coord_space` 指定 |
| `eval_lhrs.py` | LHRS-Bot、LHRS-Bot-Nova（仅 HBB） | 0–1 的 `[x1,y1,x2,y2]` |

Qwen2.5-VL / Qwen3-VL 零样本与 Qwen2.5-VL (SFT) 不需要单独脚本，直接用主评测脚本并指定
`--model_type qwen2_5_vl` 或 `qwen3_vl`。

OBB 采用严格口径：只有真正的旋转输出（带角度或四角点）才计分，面对 OBB 提示仍返回水平框记 0 分。
这正是区分"有无 OBB 能力"的依据。

```bash
# GeoChat：官方权重，以及 finetune/ 训出的 refGeo LoRA
python baselines/evaluate/eval_geochat.py --task hbb --model_path models/pretrained/geochat-7B \
    --output_dir eval_results/baselines/geochat
python baselines/evaluate/eval_geochat.py --task obb --model_path models/pretrained/geochat-7B \
    --adapter_path models/adapters/geochat_7b_refgeo_lora --output_dir eval_results/baselines/geochat_sft

# GeoGround
python baselines/evaluate/eval_geoground.py --task hbb --output_dir eval_results/baselines/geoground
python baselines/evaluate/eval_geoground.py --task obb --output_dir eval_results/baselines/geoground

# InternVL3 零样本（官方 GitHub 格式权重，ms-swift 引擎）
python baselines/evaluate/eval_internvl.py --task hbb --model_path models/pretrained/InternVL3-8B \
    --output_dir eval_results/baselines/internvl3_8b

# LLaVA-OV-1.5 零样本，以及两类微调基线（adapter 来自 finetune/）
python baselines/evaluate/eval_hf.py --model_path models/pretrained/LLaVA-OneVision-1.5-4B-Instruct \
    --task hbb --prompt_mode zeroshot --zeroshot_style qwenvl
python baselines/evaluate/eval_hf.py --model_path models/pretrained/InternVL3-8B-hf \
    --adapter_dir models/adapters/InternVL3-8B-hf_refgeo_lora --task hbb --prompt_mode trained --crop_to_patches
python baselines/evaluate/eval_hf.py --model_path models/pretrained/LLaVA-OneVision-1.5-4B-Instruct \
    --adapter_dir models/adapters/LLaVA-OneVision-1.5-4B-Instruct_refgeo_lora \
    --task hbb --prompt_mode trained --coord_space norm1 --max_pixels 802816

# LHRS-Bot-Nova
python baselines/evaluate/eval_lhrs.py --model lhrs-nova --output_dir eval_results/baselines/lhrs_nova
```

所有命令都在仓库根目录执行，`--task obb` 用法相同。每次运行会在 `--output_dir` 下新建带时间戳的
目录，写入 `<split>_<task>_predictions.jsonl`（模型原始输出、解析后的框、真值、IoU）、
`summary_<task>.json` 与一张 markdown 表。

每个基线需要各自的官方推理环境：GeoChat 用 `geochat` 包（transformers 4.31），GeoGround 用其仓库
自带的 `llava` 包，LHRS 用 `lhrs` 包。`eval_internvl.py` 直接在主 ms-swift 环境运行，`eval_hf.py`
只需要 transformers 与 peft。

## finetune/ — 基线微调

产出论文中的 "(SFT)" 各行：所有基线都用与 GeoBox-R1 相同的课程式 SFT 数据
（`data/GeoBox-R1-Data/sft/sft_curriculum_cot.jsonl`，由 [`data_pipeline/`](../data_pipeline/README_zh.md) 构建）
和相同配方——LoRA rank 16 / alpha 32、学习率 1e-4、1 个 epoch、2 卡有效 batch 16，只把坐标文本
改写成各自骨干的原生制式。

```text
prepare_hf_data.py       → refgeo_norm1000.jsonl（InternVL3，0–1000 网格）与 refgeo_norm1.jsonl（LLaVA-OV-1.5，0–1 小数）
prepare_geochat_data.py  → refgeo_geochat_native_llava.json（{<x1><y1><x2><y2>|<θ>}，0–100 网格）
train_hf_lora.py         Hugging Face 权重的 LoRA SFT（Transformers + PEFT）
train_geochat_lora.py    对官方 GeoChat 训练入口的薄封装
run_internvl3.sh         带论文超参的启动脚本；首次运行时自动生成所需数据文件
run_llava_ov15.sh
run_geochat.sh
common.sh                公共路径与辅助函数
```

```bash
MODEL_PATH=models/pretrained/InternVL3-8B-hf CUDA_DEVICES=0,1 bash baselines/finetune/run_internvl3.sh
MODEL_PATH=models/pretrained/LLaVA-OneVision-1.5-4B-Instruct CUDA_DEVICES=0,1 bash baselines/finetune/run_llava_ov15.sh
GEOCHAT_REPO=/path/to/GeoChat GEOCHAT_PYTHON=/path/to/envs/geochat/bin/python \
    CUDA_DEVICES=0,1 bash baselines/finetune/run_geochat.sh
```

所有设置都是环境变量（`BATCH_SIZE`、`GRAD_ACC`、`LEARNING_RATE`、`OUTPUT_DIR` 等；`MAX_STEPS=20`
可做冒烟测试）。adapter 输出到 `models/adapters/<checkpoint>_refgeo_lora`，用上面的
`eval_hf.py` / `eval_geochat.py --adapter_path` 命令评测。原始训练耗时约：InternVL3-2B 9.5 小时、
InternVL3-8B 15.6 小时、LLaVA-OV-1.5-4B 16.5 小时、GeoChat-7B 20.8 小时（2 卡）。

HF 训练器的依赖见 `finetune/requirements.txt`，与主环境相互独立；GeoChat 在自己的环境
（transformers 4.31、peft 0.4、deepspeed）中训练，并把官方仓库加入 `PYTHONPATH`。
