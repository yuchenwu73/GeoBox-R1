<a href="README.md">English</a> | <b>简体中文</b>

# 评测

两个脚本各自完成「加载模型 → 批量推理 → 解析框 → 计算指标」的全过程，
**一次运行同时给出三个指标**：

| 脚本 | 任务 | 指标 |
| --- | --- | --- |
| `evaluate_hbb.py` | 水平框，7 个评测集 | Acc@0.5 / Acc@0.7 / mIoU |
| `evaluate_obb.py` | 旋转框，3 个评测集 | Acc@0.5 / Acc@0.7 / mRIoU |

OBB 的旋转 IoU 用 shapely 计算多边形交并比，因此 `evaluate_obb.py` 需要
`pip install shapely`。

## 用法

```bash
# 全部评测集
CUDA_VISIBLE_DEVICES=0 python evaluation/evaluate_hbb.py \
    --model_path models/checkpoints/GeoBox-R1 --dataset all

# 单个评测集
CUDA_VISIBLE_DEVICES=0 python evaluation/evaluate_obb.py \
    --model_path models/checkpoints/GeoBox-R1 --dataset avvg_test

# 使用 norm_bbox=none 训练的 Qwen2.5-VL 检查点
CUDA_VISIBLE_DEVICES=0 python evaluation/evaluate_obb.py \
    --model_path <qwen2.5-vl-checkpoint> --model_type qwen2_5_vl \
    --coord_mode resized_absolute --dataset avvg_test
```

每次运行会新建一个带时间戳的结果目录，里面写入：

- HBB 结果默认写入 `eval_results/hbb`。
- OBB 结果默认写入 `eval_results/obb`。

```text
predictions.jsonl   逐样本的预测框、真值框与 IoU
summary.json        各数据集与宏平均的三个指标
table*.md           可直接贴进论文的 markdown 表
```

## 参数

| 参数 | 说明 |
| --- | --- |
| `--model_path` | 基础模型路径 |
| `--checkpoint_dir` | 叠加在基础模型上的 LoRA 适配器；不填则只用基础模型 |
| `--model_type` | ms-swift 模型类型。默认 `qwen3_vl`；评测 Qwen2.5-VL 基线用 `qwen2_5_vl` |
| `--dataset` | `all` 或单个测试集名 |
| `--metainfo_dir` | 测试集 jsonl 目录，默认 `data/refGeo/metainfo` |
| `--image_dir` | 图片根目录，默认 `data/refGeo/images`，支持 glob 通配 |
| `--output_dir` | 结果根目录，默认 `eval_results/hbb` 或 `eval_results/obb` |
| `--coord_mode` | 坐标约定：`norm1000`（默认，坐标量化到 0–1000）、`absolute`、`resized_absolute` |
| `--batch_size` | 覆盖各数据集的默认批大小 |
| `--attn_impl` | 注意力实现：`flash_attn`（默认，需安装 `flash-attn`）或 `sdpa` |
| `--max_samples` | 只跑前 N 条，用于快速冒烟测试 |
| `--table_format` | `text` 或 `markdown` |
| `--model_name` | 写进结果表的模型名 |

各数据集的默认批大小按图片尺寸预设（DIOR-RSVG 800×800 用 60，AVVG 大图用 12），按整张
40 GB 显卡空闲估算；显卡更小或与他人共用时用 `--batch_size` 调小。整批失败（通常是显存不足）
会自动改为逐条重试；仍然失败的样本按 0 计入，并在 `predictions.jsonl` 里带 `error` 字段，
逐数据集的汇总会打印失败条数，因此指标不会悄悄丢样本。

## 坐标约定

`norm1000` 与训练时一致：所有坐标量化到 `[0, 1000]`。评测其他模型时，如果它输出的是
原图像素坐标，用 `--coord_mode absolute`；如果是缩放后图像的像素坐标，用
`--coord_mode resized_absolute`。本仓库的 Qwen2.5-VL 基线使用后者。
