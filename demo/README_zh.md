<a href="README.md">English</a> | <b>简体中文</b>

# GeoBox-R1 演示

GeoBox-R1 的 Gradio 演示：上传一张遥感图像，用自然语言描述目标，模型返回水平框（HBB）
或旋转框（OBB）。界面为中英双语。

## 启动

```bash
cd demo
bash run_demo.sh
```

默认使用 GPU 0、端口 7860，模型取仓库根下的 `models/checkpoints/GeoBox-R1`。

```bash
# 指定 GPU / 端口
GEOBOX_GPU=1 GEOBOX_PORT=8000 bash run_demo.sh

# 换模型：本地路径，或直接用 Hugging Face 仓库名
GEOBOX_MODEL=yuchenwu73/GeoBox-R1 bash run_demo.sh

# 生成临时公网链接（https://<随机串>.gradio.live）
GEOBOX_SHARE=1 bash run_demo.sh

# 没有 flash-attn 时；以及指定解释器（默认用 PATH 里的 python）
GEOBOX_ATTN=sdpa GEOBOX_PYTHON=/path/to/envs/geobox-r1/bin/python bash run_demo.sh
```

在远程服务器上跑时，用 SSH 端口转发从本地访问更稳定，地址也固定：

```bash
ssh -N -L 7860:localhost:7860 <user>@<server>
# 然后本地浏览器打开 http://localhost:7860
```

Gradio 免费版的 `gradio.live` 链接每次启动都是随机的，无法固定。

## 内置样例

演示内置 5 个精选样例，覆盖 2 个水平框与 3 个旋转框任务：

| # | 数据集 | 任务 | 看点 |
|---|--------|------|------|
| 1 | DIOR-RSVG | HBB | 红色车辆左上方的网球场，考察空间关系推理 |
| 2 | VRSBench | HBB | 最右侧的飞机，目标清晰 |
| 3 | GeoChat | OBB | 约 45° 斜向的长条形货船，旋转框贴合船体 |
| 4 | AVVG | OBB | 斜向的车辆，旋转框贴合车身 |
| 5 | VRSBench | OBB | 约 45° 斜向的车辆，旋转框与水平框差异明显 |

## 文件

```text
demo/
├── app.py                  # Gradio 页面
├── inference.py            # GeoBox-R1 推理封装（提示词与评测脚本逐字一致）
├── visualize.py            # GT / Pred 框绘制与标签防重叠
├── prepare_examples.py     # 从 refGeo 重新生成样例
├── selftest.py             # 批量自检
├── examples_manifest.json  # 样例元数据
├── examples/               # 样例图片
└── run_demo.sh             # 启动脚本
```

## 自检

在不开界面的情况下跑完 5 个样例，输出 IoU / RIoU 并把可视化写到 `selftest_out/`：

```bash
CUDA_VISIBLE_DEVICES=0 python selftest.py
```

## 重新生成样例

`prepare_examples.py` 从 refGeo 的测试集里挑样例并复制图片。需要先按仓库根
[README](../README_zh.md#数据) 的说明准备好 `data/refGeo/`：

```bash
REFGEO_ROOT=../data/refGeo python prepare_examples.py
```
