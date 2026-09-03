<div align="center">

<a href="README.md">English</a> | <b>简体中文</b>

# GeoBox-R1：面向统一框级遥感视觉定位的课程式监督微调与几何强化学习

Chenxi Lan\*, Yuchen Wu\*, Minghang Zhou, Tianyu Li, Zhihao Qiu, Guoqing Wang<sup>†</sup>

电子科技大学

<sup>\*</sup>共同第一作者 &nbsp;&nbsp; <sup>†</sup>通讯作者

*AAAI 2027 投稿中*

[![项目主页](https://img.shields.io/badge/%E9%A1%B9%E7%9B%AE-%E4%B8%BB%E9%A1%B5-1d4ed8)](https://yuchenwu73.github.io/geobox-r1/)
[![模型](https://img.shields.io/badge/%F0%9F%A4%97-GeoBox--R1-ffc107)](https://huggingface.co/yuchenwu73/GeoBox-R1)

</div>

---

## 简介

遥感视觉定位（RSVG）的任务是：给定一句自然语言描述，在航拍或卫星影像中定位它所指的目标。
多模态大模型（MLLM）让**统一的框级定位**成为可能——同一个模型既输出水平框（HBB）也输出
旋转框（OBB）——但有两个障碍：

- 通用 MLLM **无法稳定输出 OBB**。即使提示词明确要求旋转框，它们也常常退回到忽略目标朝向的
  水平框。
- 用一个模型同时训练两个任务并不容易，因为 HBB 与 OBB 对**语义定位粒度和几何精度的要求不同**。

**GeoBox-R1** 以 Qwen3-VL-4B-Instruct 为基座，用两阶段方案同时解决这两点：

| 阶段 | 做法 |
| --- | --- |
| **一、课程式 SFT** | 训练数据按由易到难排列——HBB 定位 → OBB 定位 → HBB-to-OBB 思维链——而不是随机打乱。在一次训练中建立由粗到精的定位能力。 |
| **二、几何强化学习（GDPO）** | 分通道奖励解耦归一化的策略优化，由两个规则化几何奖励驱动：**旋转 IoU** 与**自适应 Wasserstein 距离**——后者在 RIoU 趋近于零而失去梯度时依然提供有效信号。全程不需要学习式奖励模型。 |

在 7 个 HBB 与 3 个 OBB 基准上，**4B 参数**的 GeoBox-R1 在每项指标的宏平均上都超过了 7B–8B 的
现有最优模型，且在更严格的 Acc@0.7 阈值上领先幅度最大。OBB 上它在每个评测集的每项指标都领先；
HBB 上逐基准互有胜负：GeoGround 在 DIOR-RSVG 和 GeoChat\* 的多数指标上领先，GeoBox-R1 在 RSVG
和 AVVG 的全部指标以及 VRSBench\* 的两个准确率阈值上领先。

## 结果

下表为各评测集的宏平均，只列每组最强的基线；含全部基线的完整表格见[项目主页](https://yuchenwu73.github.io/geobox-r1/)。

**HBB 定位**（7 个评测集）

| 模型 | 参数量 | Acc@0.5 | Acc@0.7 | mIoU |
| --- | :-: | :-: | :-: | :-: |
| Qwen3-VL | 8B | 43.70 | 28.53 | 39.19 |
| Qwen2.5-VL (SFT) | 3B | 49.43 | 32.57 | 43.08 |
| GeoGround | 7B | 52.35 | 35.02 | 46.80 |
| InternVL3 (SFT) | 8B | 55.10 | 37.33 | 47.78 |
| GeoBox-R1 (SFT) | 4B | 57.17 | 40.33 | 48.82 |
| **GeoBox-R1 (SFT + GDPO)** | **4B** | **58.78** | **42.22** | **50.39** |

**OBB 定位**（3 个评测集）

| 模型 | 参数量 | Acc@0.5 | Acc@0.7 | mRIoU |
| --- | :-: | :-: | :-: | :-: |
| Qwen2.5-VL (SFT) | 3B | 35.72 | 17.61 | 31.86 |
| InternVL3 (SFT) | 8B | 35.74 | 17.05 | 32.58 |
| GeoGround | 7B | 41.96 | 19.81 | 36.96 |
| GeoBox-R1 (SFT) | 4B | 43.11 | 24.40 | 36.85 |
| **GeoBox-R1 (SFT + GDPO)** | **4B** | **47.32** | **27.55** | **39.85** |

GDPO 只用 OBB 样本训练，却让**两个任务同时上升**——相对 SFT 检查点，HBB Acc@0.5 提升 1.61，
OBB 提升 4.21。

**GeoBox-R1 (SFT + GDPO) 逐评测集结果**——即 `evaluation/evaluate_hbb.py` 与
`evaluate_obb.py` 对已发布检查点打印的数字。

| 评测集 | 任务 | Acc@0.5 | Acc@0.7 | mIoU / mRIoU |
| --- | :-: | :-: | :-: | :-: |
| DIOR-RSVG test | HBB | 76.61 | 64.78 | 67.50 |
| DIOR-RSVG val | HBB | 75.11 | 63.13 | 66.57 |
| RSVG test | HBB | 51.26 | 32.60 | 42.99 |
| RSVG val | HBB | 48.38 | 31.97 | 42.00 |
| GeoChat\* | HBB | 61.13 | 35.49 | 51.25 |
| VRSBench\* | HBB | 66.84 | 41.16 | 55.74 |
| AVVG | HBB | 32.14 | 26.39 | 26.68 |
| GeoChat\* | OBB | 60.56 | 35.19 | 48.92 |
| VRSBench\* | OBB | 56.61 | 30.55 | 49.43 |
| AVVG | OBB | 24.79 | 16.89 | 21.18 |

\* GeoChat 与 VRSBench 采用 refGeo 重新划分的版本（去掉了与 DIOR-RSVG val/test 图片重叠的样本），
与 GeoGround 一致。所有基线均由我们用各模型专属的定位提示词和解析器重新评测。

## 仓库结构

```text
GeoBox-R1/
├── evaluation/          # 评测：一次跑出 Acc@0.5、Acc@0.7 与 mIoU/mRIoU
│   ├── evaluate_hbb.py
│   └── evaluate_obb.py
├── training/            # SFT、GDPO、vLLM rollout、LoRA 合并
│   ├── sft.sh
│   ├── gdpo.sh
│   ├── gdpo_fixed_tau.sh          # 固定 τ 的对照
│   ├── gdpo_qwen2_5vl.sh          # 同一套流程的 Qwen2.5-VL 版本
│   ├── rollout_vllm.sh
│   ├── merge_lora.sh
│   ├── reward_plugin_qwen3vl.py   # RIoU + 自适应 Wasserstein 奖励
│   └── reward_plugin_qwen2_5vl.py # 同一套奖励的 Qwen2.5-VL 坐标制式版本
├── data_pipeline/       # 构建 refGeo 的 SFT / RL / 消融数据
├── visualization/       # 定性结果绘图与 τ 分析
├── demo/                # Gradio 演示
├── baselines/           # 基线模型的评测与微调
└── tests/               # 单元测试，不需要 GPU
```

每个目录都有自己的 README，中英文各一份。

另有两个目录需要自行准备，不纳入版本管理（见[环境准备](#环境准备)）：

```text
data/
├── refGeo/            metainfo/ 与 images/——评测用，也是训练数据的来源
└── GeoBox-R1-Data/    训练数据 sft/ 与 rl/，由 data_pipeline/ 在本地构建
models/
├── pretrained/        Qwen3-VL-4B-Instruct
└── checkpoints/       GeoBox-R1（已发布）；GeoBox-R1-SFT 由第一阶段训练产生
```

## 环境准备

```bash
git clone https://github.com/yuchenwu73/GeoBox-R1.git
cd GeoBox-R1

conda create -n geobox-r1 python=3.10 -y
conda activate geobox-r1
pip install -r requirements.txt
# 所有训练脚本与评测脚本默认使用 Flash-Attention 2。请从
# https://github.com/Dao-AILab/flash-attention/releases 选择与你的 torch / CUDA / Python 匹配的 wheel，
# 下面这个对应本仓库固定的 torch 2.8 / CUDA 12 / Python 3.10：
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp310-cp310-linux_x86_64.whl

python -m unittest discover -s tests   # 检查各脚本，不需要 GPU 和模型
```

训练与评测基于 [ms-swift](https://github.com/modelscope/ms-swift) 4.0.2。
以上依赖已在全新的 Python 3.10 环境中验证可以直接安装，装完后单元测试、数据流水线、评测脚本和
demo 均能运行。没有 `flash-attn` 时，评测可加 `--attn_impl sdpa`，demo 用 `GEOBOX_ATTN=sdpa`
启动；训练脚本的 `--packing` / `--padding_free` 依赖它。

### 数据

训练数据不对外分发。请用 [data_pipeline](data_pipeline/) 从 refGeo 标注构建：流程是确定性的
（seed 42），能重新生成已发布模型实际训练所用的文件，目录结构如下：

```text
data/GeoBox-R1-Data/
├── sft/
│   ├── sft_curriculum_cot.jsonl      # 主训练集 —— 课程顺序，含 CoT
│   ├── sft_curriculum_no_cot.jsonl
│   ├── sft_mixed_cot.jsonl
│   └── sft_mixed_no_cot.jsonl        # 以上三个是消融对照组
└── rl/
    └── rl_obb_20pct.jsonl            # GDPO 用的 20% OBB 子集
```

图片来自各原始基准，本仓库不二次分发；请参照 [refGeo](https://github.com/zytx121/GeoGround)
准备后按下列结构放置：

```text
data/refGeo/
├── metainfo/          # 各测试集的 *.jsonl
└── images/
    ├── DIOR-RSVG/
    ├── RSVG/
    ├── GeoChat/
    ├── VRSBench/
    └── AVVG/
```

也可以用 `--metainfo_dir` / `--image_dir` 指定其他位置，部分脚本支持 `REFGEO_ROOT` 环境变量。
想从零重建训练数据，见 [`data_pipeline/`](data_pipeline/README_zh.md)。

### 模型

| 检查点 | 说明 |
| --- | --- |
| [`GeoBox-R1`](https://huggingface.co/yuchenwu73/GeoBox-R1) | 最终模型，课程式 SFT + GDPO。复现论文数字请用这个。 |

第一阶段检查点不对外分发：先跑 `training/sft.sh`，再执行 `training/merge_lora.sh sft`，
即得到 `models/checkpoints/GeoBox-R1-SFT`，第二阶段从它开始。

## 评测

两个脚本都会**在一次运行中给出 Acc@0.5、Acc@0.7 以及 mIoU（HBB）/ mRIoU（OBB）**，
并把 `predictions.jsonl` 与 `summary.json` 写入带时间戳的结果目录。

```bash
# HBB —— 7 个评测集
CUDA_VISIBLE_DEVICES=0 python evaluation/evaluate_hbb.py \
    --model_path models/checkpoints/GeoBox-R1 \
    --dataset all

# OBB —— 3 个评测集
CUDA_VISIBLE_DEVICES=0 python evaluation/evaluate_obb.py \
    --model_path models/checkpoints/GeoBox-R1 \
    --dataset all
```

常用参数：

| 参数 | 含义 |
| --- | --- |
| `--dataset` | `all`，或单个测试集（`dior_rsvg_test`、`rsvg_test`、`geochat_test`、`vrsbench_test`、`avvg_test` 等） |
| `--checkpoint_dir` | 叠加在 `--model_path` 之上的 LoRA 适配器 |
| `--model_type` | ms-swift 模型类型；默认 `qwen3_vl`，评测基线时用 `qwen2_5_vl` |
| `--coord_mode` | 坐标约定：`norm1000`（默认）、`absolute`、`resized_absolute` |
| `--batch_size` | 覆盖各数据集的默认批大小 |
| `--table_format` | `text` 或 `markdown` |

评测本仓库的 Qwen2.5-VL 基线时，需要同时传入 `--model_type qwen2_5_vl` 和
`--coord_mode resized_absolute`。

## 训练

```bash
# 第一阶段 —— 课程式 SFT
bash training/sft.sh

# 第二阶段 —— GDPO。先起 vLLM rollout 服务，再启动训练
bash training/rollout_vllm.sh
bash training/gdpo.sh

# 把 LoRA 适配器合并成独立检查点
bash training/merge_lora.sh sft     # → GeoBox-R1-SFT
bash training/merge_lora.sh gdpo    # → GeoBox-R1
```

课程顺序体现在数据本身，而不在训练脚本里 —— `sft.sh` 读的是 `sft_curriculum_cot.jsonl`，
训练时不能打乱。一旦打乱，跑出来的就是消融里的混合基线。

`training/reward_plugin_qwen3vl.py` 实现了两个几何奖励。自适应尺度为 $\tau = \tau_c \cdot \sqrt{\mathrm{Tr}(\Sigma_g)}$，
用于把奖励灵敏度按目标尺寸归一化；`training/gdpo_fixed_tau.sh` 是固定 τ 的对照。
`reward_plugin_qwen2_5vl.py` 是同一套奖励的 Qwen2.5-VL 版本，两者坐标制式不同，
对应的启动脚本是 `training/gdpo_qwen2_5vl.sh`。

第一阶段用 LoRA rank 16 / alpha 32，冻结视觉编码器与 merger，学习率 `1e-4`，
在 2 张 RTX 4090 上训练 1 个 epoch。第二阶段在 20% 的 OBB 子集上进行（96,791 条中取 19,357 条）。

## 演示

```bash
cd demo
bash run_demo.sh
```

上传一张遥感图像，输入目标描述，选择 HBB 或 OBB 即可。演示内置 5 个精选样例。
GPU 与端口等选项见 [`demo/README.md`](demo/README.md)。

## 引用

```bibtex
@misc{geoboxr1,
  title  = {GeoBox-R1: Curriculum-Guided SFT and Geometric RL for
            Unified Box-Level Remote Sensing Visual Grounding},
  author = {Lan, Chenxi and Wu, Yuchen and Zhou, Minghang and
            Li, Tianyu and Qiu, Zhihao and Wang, Guoqing},
  year   = {2026},
  url    = {https://yuchenwu73.github.io/geobox-r1/},
  note   = {Preprint}
}
```

## 致谢

本工作基于 [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) 与
[ms-swift](https://github.com/modelscope/ms-swift) 构建，refGeo 的评测设定沿用
[GeoGround](https://github.com/zytx121/GeoGround)。感谢 DIOR-RSVG、RSVG、GeoChat、
VRSBench、AVVG 各数据集作者的开放共享。

## 许可

代码以 MIT 协议发布。模型权重遵循 CC BY-NC 4.0；自行构建的训练数据受各原始基准自身许可约束。
