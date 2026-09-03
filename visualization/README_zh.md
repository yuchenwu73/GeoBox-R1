<a href="README.md">English</a> | <b>简体中文</b>

# 可视化与分析

## 定性结果绘图

| 脚本 | 用途 |
| --- | --- |
| `compare_testsets.py` | 批量跨测试集的多模型对比图，论文图 3、图 4 的来源 |
| `export_sample.py` | 单样本上逐模型导出预测框，复用 `compare_testsets.py` 写下的 cache |
| `plot_obb_boxes.py` | 只画框的 OBB 真值 vs 预测对比，适合放大看贴合度 |
| `infer_and_plot.py` | 自动推理并可视化，支持 HBB 与 OBB |

坐标默认按 `norm1000` 反归一化到原图尺寸，与评测脚本一致。所有命令在仓库根目录执行，
结果写到 `output/visualizations/`。

```bash
# 单张图，同时出 HBB 与 OBB；--gpu 可选，只有给出时才会改 CUDA_VISIBLE_DEVICES
python visualization/infer_and_plot.py --image path/to/scene.png --query "the ship at the pier" --mode both --gpu 0

# 对所有模型、所有评测集推理，然后生成五栏对比图
python visualization/compare_testsets.py --task obb --models all

# 从上一步的 cache 里导出单个样本（默认取该任务默认测试集的第一个样本）
python visualization/export_sample.py --task obb --dataset vrsbench_test --question-id 12

# 只画框，坐标直接给出
python visualization/plot_obb_boxes.py --image-path scene.png --gt-obb "[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]" --pred-obb "[[...]]"
```

## τ 的取值分析

自适应 Wasserstein 奖励里的尺度 `τ = τ_c · sqrt(Tr(Σ_g))` 需要定 `τ_c`：

- `analyze_tau_adaptive.py` 统计 RL 训练数据里 OBB 的尺寸与长宽比分布，
  据此给出 `τ_c` 的建议区间。
- `analyze_tau_fixed.py` 对照固定 τ 的表现，说明为什么要让 τ 随目标尺寸缩放——
  固定 τ 下，同样的相对定位误差在大目标上会得到系统性更低的奖励。

结论用于 `training/gdpo.sh`；`training/gdpo_fixed_tau.sh` 保留固定 τ 的对照实验。

两个分析脚本都可从任意工作目录运行，数据路径统一按仓库根目录解析为
`data/GeoBox-R1-Data/rl/rl_obb_20pct.jsonl`。
