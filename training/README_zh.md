<a href="README.md">English</a> | <b>简体中文</b>

# 训练

两阶段流程，均基于 [ms-swift](https://github.com/modelscope/ms-swift) 4.0.2。

## 第一阶段：课程式 SFT

```bash
bash training/sft.sh
```

课程顺序体现在**数据本身**：训练样本按 HBB 定位 → OBB 定位 → HBB-to-OBB 思维链排列。
`sft.sh` 同时关闭数据集和 DataLoader shuffle，保持原始顺序。

配置：LoRA rank 16 / alpha 32，冻结视觉编码器与 merger，学习率 `1e-4`，1 个 epoch，
2 × RTX 4090 (24 GB)。数据 161,692 条（32,638 HBB + 96,791 OBB + 32,263 CoT）。

## 第二阶段：GDPO

先起 vLLM rollout 服务，再启动训练（两个终端）：

```bash
bash training/rollout_vllm.sh   # 终端 1
bash training/gdpo.sh           # 终端 2
```

`rollout_vllm.sh` 监听 7772 端口，`gdpo.sh` 里的 rollout 地址需与之对应。

奖励实现在 `reward_plugin_qwen3vl.py`：

- **旋转 IoU**：对旋转框的直接重叠监督。
- **自适应 Wasserstein 距离**：把预测框与真值框视作二维高斯，用 2-Wasserstein 距离度量差异，
  再经 `r = τ / (τ + W₂)` 转成有界奖励。尺度 `τ = τ_c · sqrt(Tr(Σ_g))` 随目标尺寸自适应——
  固定 τ 会让大目标系统性地拿到更低奖励。这一项在 RIoU 趋近于零、失去梯度时仍然有效。

两项奖励等权（各 0.5），在 20% 的 OBB 子集上训练（96,791 条中取 19,357 条）。

`gdpo_fixed_tau.sh` 是固定 τ 的对照，复用同一个 rollout 服务，两个实验依次跑即可；
`visualization/analyze_tau_fixed.py` 与 `analyze_tau_adaptive.py` 用于确定 τ_c 的取值。

### Qwen2.5-VL

`reward_plugin_qwen2_5vl.py` 处理 Qwen2.5-VL 的缩放图像像素坐标；Qwen3-VL 插件则使用
`norm1000`。前者用 `*_qwen2_5` 后缀注册奖励函数，避免名称冲突。

```bash
bash training/rollout_vllm.sh qwen2_5
bash training/gdpo_qwen2_5vl.sh
```

Qwen2.5-VL 的 rollout 服务使用 7773 端口，与对应 GDPO 启动脚本一致。

## 合并 LoRA

```bash
bash training/merge_lora.sh sft    # SFT 适配器  → GeoBox-R1-SFT
bash training/merge_lora.sh gdpo   # GDPO 适配器 → GeoBox-R1
bash training/merge_lora.sh gdpo-qwen2_5
```

合并脚本会从版本化运行目录中选择最新 adapter。目录不同时可用 `ADAPTER_ROOT=`、
`ADAPTER=` 或 `OUTPUT=` 覆盖；评测与演示都使用合并后的检查点。

## 路径

所有启动脚本都会先切换到**仓库根目录**，脚本内的路径（`models/...`、`data/GeoBox-R1-Data/...`、
`training/reward_plugin_*.py`）均相对仓库根目录，因此上面的命令可以在任意目录下执行。
这一点对数据很重要：训练 JSONL 里的图片路径同样相对仓库根目录
（`data/refGeo/images/<子集>/<文件名>`，由 `data_pipeline/` 写出），而 ms-swift 按当前工作目录打开图片。
`sft.sh` 与 `gdpo*.sh` 的 `--dataset` 指向这些流水线产物；若自行拼装 JSONL，请保证图片路径从仓库根目录可达。

`merge_lora.sh` 的覆盖变量（`ADAPTER_ROOT=`、`ADAPTER=`、`OUTPUT=`、`BASE_MODEL=`）遵循同一规则：
相对路径从仓库根目录解析。
