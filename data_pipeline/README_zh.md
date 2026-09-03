<a href="README.md">English</a> | <b>简体中文</b>

# 数据流水线

用四步从 refGeo 标注构建 GeoBox-R1 的训练数据。所有命令在仓库根目录执行：脚本读取
`data/refGeo/metainfo/*_train.jsonl`，把中间产物放在旁边的 `data/refGeo/SFT`、`data/refGeo/RL`，
最终训练集写到 `data/GeoBox-R1-Data/`。训练数据不对外分发；流程是确定性的（seed 42），
能复现已发布模型实际训练所用的文件。

## 流程

```text
data/refGeo/metainfo/*_train.jsonl
  │
  ├─ 1. build_hbb.py      RSVG、DIOR-RSVG          →  SFT/{RSVG,DIOR-RSVG}_HBB_train.jsonl
  └─ 2. build_obb_cot.py  GeoChat、VRSBench、AVVG  →  SFT/<Subset>_OBB_train.jsonl   每个子集的 75%
                                                      SFT/<Subset>_CoT_train.jsonl   其余 25%，HBB→OBB 的 CoT
          │
          ├─ 3. build_sft.py   HBB + OBB + CoT        →  data/GeoBox-R1-Data/sft/sft_<config>.jsonl
          └─ 4. build_rl.py    OBB 部分 → RL 格式      →  data/GeoBox-R1-Data/rl/rl_obb_20pct.jsonl
```

```bash
python data_pipeline/build_hbb.py
python data_pipeline/build_obb_cot.py
python data_pipeline/build_sft.py --config all   # 默认只生成 curriculum_cot
python data_pipeline/build_rl.py
python data_pipeline/check_format.py             # 可选：打印几条编码后的样本
```

所有脚本都接受 `--refgeo_root`（默认 `data/refGeo`）；第 3、4 步还接受 `--output_dir`
（默认 `data/GeoBox-R1-Data`）和 `--seed`（默认 42）。记录里的图片路径写成
`<refgeo_root>/images/<Subset>/<file>`，训练脚本以仓库根目录为工作目录打开它们。
`check_format.py` 需要 ms-swift 和基座模型，其余脚本只依赖 Pillow 与 tqdm。

## 主模型用哪个文件

`sft_curriculum_cot.jsonl`。`build_sft.py` 把同样的 161,692 条样本按四种方式组织，对应论文里
数据组织方式 × CoT 监督的 2×2 消融：


| 配置                | 顺序                      | CoT | 对应                                        |
| --------------------- | --------------------------- | :---: | --------------------------------------------- |
| `curriculum_cot`    | HBB → OBB → CoT，不打乱 | ✓ | **课程式 + CoT——主训练集**                |
| `curriculum_no_cot` | HBB → OBB，不打乱        | ✗ | 课程式、无 CoT（CoT 样本转成普通 OBB 样本） |
| `mixed_cot`         | 全局打乱                  | ✓ | Mixed + CoT                                 |
| `mixed_no_cot`      | 全局打乱                  | ✗ | Mixed、无 CoT                               |

课程就是文件本身的顺序，因此 `training/sft.sh` 关闭了数据集和 dataloader 的打乱。
`mixed_*` 只用于消融对照。

`*_no_cot` 两个消融文件里，由 CoT 转换而来的记录沿用原始训练文件中的提示词原文（结尾代码围栏前多一个空行），
因此 `--config all` 能逐条复现原始的四个文件。

RL 阶段用 `rl_obb_20pct.jsonl`：按 AVVG、GeoChat、VRSBench 的顺序，用固定种子从每个 OBB 部分抽 20%。
RL 记录额外带 `oriented_bbox`、`image_width`、`image_height`，奖励函数用它们把真值映射到模型的
`norm1000` 坐标空间。

## 记录格式

```json
{"messages": [{"role": "user", "content": "<image>Locate the instance that matches the description: [<ref-object>]. ..."},
              {"role": "assistant", "content": "```json\n[\n\t{\"horizontal_bbox\": <bbox>}\n]\n```"}],
 "images": ["data/refGeo/images/DIOR-RSVG/00001.jpg"],
 "objects": {"ref": ["the tennis court on the upper left"], "bbox": [[287, 146, 408, 398]]},
 "origin_dataset": "DIOR-RSVG"}
```

`<ref-object>` 和 `<bbox>` 是 ms-swift 的占位符：指代表达和框保存在 `objects` 里，训练时才渲染进文本
（`QWENVL_BBOX_FORMAT=new` 输出 norm1000 坐标）。OBB 记录的 `objects.bbox` 是四个角点；CoT 记录是
`[hbb, p1, p2, p3, p4]`，因为它的回答先消费水平框。所有坐标都是原图像素，四舍五入取整。

## 数据量


| 数据集    |        HBB |        OBB |        CoT |        合计 |
| ----------- | -----------: | -----------: | -----------: | ------------: |
| DIOR-RSVG |     27,133 |          0 |          0 |      27,133 |
| RSVG      |      5,505 |          0 |          0 |       5,505 |
| GeoChat   |          0 |     47,912 |     15,971 |      63,883 |
| VRSBench  |          0 |     29,017 |      9,672 |      38,689 |
| AVVG      |          0 |     19,862 |      6,620 |      26,482 |
| **合计**  | **32,638** | **96,791** | **32,263** | **161,692** |

RL 子集保留 GeoChat 9,582 条、VRSBench 5,803 条、AVVG 3,972 条，共 19,357 条。
