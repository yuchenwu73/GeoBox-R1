#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gradio interface for GeoBox-R1 visual grounding."""

import json
import os
import threading

import gradio as gr

import inference
import visualize

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLES_DIR = os.path.join(HERE, "examples")
MANIFEST_PATH = os.path.join(HERE, "examples_manifest.json")

TASK_HBB = "HBB · Horizontal box · 水平框"
TASK_OBB = "OBB · Oriented box · 旋转框"
TASK_MAP = {TASK_HBB: "hbb", TASK_OBB: "obb"}

with open(MANIFEST_PATH, encoding="utf-8") as f:
    MANIFEST = json.load(f)

# Map cached example filenames back to their metadata and ground truth.
BY_BASENAME = {os.path.basename(m["image_file"]): m for m in MANIFEST}

model = inference.get_model()


def _lookup_sample(image_path: str):
    """Return bundled-example metadata for an image path, if available."""
    if not image_path:
        return None
    base = os.path.basename(image_path)
    if base in BY_BASENAME:
        return BY_BASENAME[base]
    # Gradio may rename cached uploads, so fall back to matching the original stem.
    for name, entry in BY_BASENAME.items():
        stem = os.path.splitext(name)[0]
        if stem in base:
            return entry
    return None


def _row(key, value, accent=False):
    cls = "rd-val accent" if accent else "rd-val"
    return f'<div class="rd-row"><span class="rd-key">{key}</span><span class="{cls}">{value}</span></div>'


def render_panel(result, iou, sample, task_code):
    if result is None:
        return (
            '<div class="readout">'
            '<div class="rd-head"><span class="dot"></span>STANDBY · 系统就绪</div>'
            '<div class="rd-hint">Load an example or upload an aerial image, describe the target, '
            'choose HBB / OBB and press <b>LOCATE</b>.<br>'
            '载入样例或上传一张遥感影像，输入目标描述，选择 HBB / OBB 后点击 <b>执行定位</b>。</div>'
            "</div>"
        )

    accent_hex = "#0e7490" if task_code == "hbb" else "#b45309"
    if result["parsed_ok"]:
        status = "TARGET LOCKED · 已锁定目标"
        status_color = accent_hex
    else:
        status = "NO TARGET · 未解析到坐标"
        status_color = "#c0392b"

    rows = []
    rows.append(f'<div class="rd-head" style="color:{status_color}">'
                f'<span class="dot" style="background:{status_color};box-shadow:0 0 0 3px {status_color}22"></span>'
                f"{status}</div>")

    task_label = TASK_HBB if task_code == "hbb" else TASK_OBB
    rows.append(_row("Task · 任务", task_label, accent=True))
    rows.append(_row("Image size · 影像尺寸", f'{result["image_size"][0]} × {result["image_size"][1]} px'))
    rows.append(_row("Latency · 推理耗时", f'{result["latency"]:.2f} s'))

    if result["parsed_ok"]:
        norm = result["norm_coords"]
        if task_code == "hbb":
            px = result["bbox_px"]
            rows.append(_row("Normalized 0–1000 · 归一化", f"[{', '.join(str(int(v)) for v in norm)}]"))
            rows.append(_row("Pixels x1y1x2y2 · 像素坐标", f"[{', '.join(f'{v:.0f}' for v in px)}]", accent=True))
        else:
            pts_norm = " ".join(f"({int(p[0])},{int(p[1])})" for p in norm)
            pts_px = " ".join(f"({p[0]:.0f},{p[1]:.0f})" for p in result["poly_px"])
            rows.append(_row("Normalized corners · 归一化四点", pts_norm))
            rows.append(_row("Pixel corners · 像素四点", pts_px, accent=True))

    if iou is not None:
        thr = 0.5
        hit = iou >= thr
        hit_txt = "Hit ✔ 命中 (Acc@0.5)" if hit else "Miss ✘ 未命中"
        hit_color = "#15803d" if hit else "#c0392b"
        metric = "IoU" if task_code == "hbb" else "Rotated IoU"
        rows.append(f'<div class="rd-row"><span class="rd-key">{metric} vs GT</span>'
                    f'<span class="rd-val" style="color:{hit_color};font-weight:600">{iou:.4f} · {hit_txt}</span></div>')
    elif sample is None:
        rows.append(_row("Ground truth · 真值", "user upload, no GT · 用户上传，无真值对照"))

    raw = (result["raw_output"] or "").strip()
    raw_disp = raw if len(raw) < 600 else raw[:600] + " …"
    rows.append(f'<div class="rd-raw"><div class="rd-key">Raw model output · 模型原始输出</div>'
                f'<pre>{_esc(raw_disp)}</pre></div>')

    if sample is not None:
        rows.append(f'<div class="rd-note"><span class="rd-note-tag">{sample["dataset_label"]}</span>'
                    f'{sample["note"]}</div>')

    return f'<div class="readout">{"".join(rows)}</div>'


def _esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;"))


def _blank_result(text: str = "AWAITING INPUT"):
    """Return the placeholder image and standby panel."""
    return visualize.make_placeholder(text), render_panel(None, None, None, "hbb")


def _running_panel(task_code: str):
    name = task_code.upper()
    return (
        '<div class="readout">'
        '<div class="rd-head"><span class="dot"></span>RUNNING · 正在推理</div>'
        f'<div class="rd-hint">Running {name} grounding. The previous result was cleared; '
        'large images are downscaled on the server before display.<br>'
        f'正在执行 {name} 定位。旧结果已清空；大图会在服务器端压缩后再返回。</div>'
        '</div>'
    )


def _resize_for_ui(image, max_side: int = 1600):
    """Bound output size to keep browser encoding and transfer responsive."""
    w, h = image.size
    long_side = max(w, h)
    if long_side <= max_side:
        return image
    scale = max_side / float(long_side)
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    resampling = getattr(__import__("PIL").Image, "Resampling", None)
    method = resampling.LANCZOS if resampling else 1
    return image.resize(new_size, method)


def run_inference(image_path, question, task_label):
    """Yield a running state before the final annotated result."""
    if task_label not in TASK_MAP:
        raise ValueError(f"Unsupported task label: {task_label!r}")
    task_code = TASK_MAP[task_label]
    if not image_path:
        yield _blank_result("NO IMAGE")
        return
    if not question or not question.strip():
        gr.Warning("Please describe the target in natural language. 请输入目标的自然语言描述。")
        yield _blank_result("AWAITING INPUT")
        return

    yield visualize.make_placeholder("RUNNING"), _running_panel(task_code)

    sample = _lookup_sample(image_path)
    result = model.infer(image_path, question.strip(), task_code)

    iou = None
    if sample is not None and result["parsed_ok"]:
        if task_code == "hbb" and result["bbox_px"] and sample.get("gt_bbox"):
            iou = inference.iou_hbb(result["bbox_px"], sample["gt_bbox"])
        elif task_code == "obb" and result["poly_px"] and sample.get("gt_poly"):
            iou = inference.iou_obb(result["poly_px"], sample["gt_poly"])

    gt_bbox = sample.get("gt_bbox") if sample else None
    gt_poly = sample.get("gt_poly") if sample else None
    annotated = visualize.draw_result(image_path, result, gt_bbox=gt_bbox, gt_poly=gt_poly, iou=iou)
    annotated = _resize_for_ui(annotated)
    panel = render_panel(result, iou, sample, task_code)
    yield annotated, panel


def on_example_select(image_path):
    """Populate bundled-example inputs and clear the previous result."""
    sample = _lookup_sample(image_path)
    if sample is None:
        # Preserve query and task for user uploads while clearing stale output.
        return gr.update(), gr.update(), * _blank_result("AWAITING INPUT")
    task_label = TASK_HBB if sample["recommended_task"] == "hbb" else TASK_OBB
    return sample["question"], task_label, * _blank_result("AWAITING INPUT")


CSS = """
@import url('https://fonts.googleapis.com/css2?family=Chakra+Petch:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root{
  --bg:#edf6fb; --surface:#f8fbff; --panel:#ffffff;
  --ink:#202a33; --ink-soft:#46525d; --muted:#64748b;
  --teal:#0e7490; --amber:#b45309; --green:#15803d; --red:#c0392b;
  --line:rgba(32,42,51,0.13); --line-2:rgba(32,42,51,0.26);
  --mono:'IBM Plex Mono',ui-monospace,'DejaVu Sans Mono',monospace;
  --disp:'Chakra Petch','Rajdhani',var(--mono);
}

/* Keep the application centered on wide screens. */
.gradio-container{
  max-width:1280px !important; margin:0 auto !important;
  background:transparent !important; padding:18px 20px 30px !important;
}
body, gradio-app{
  /* Subtle survey-grid background. */
  background:
    radial-gradient(900px 420px at 12% -8%, rgba(14,116,144,0.11), transparent 62%),
    radial-gradient(900px 460px at 88% 4%, rgba(59,130,246,0.075), transparent 58%),
    repeating-linear-gradient(0deg, rgba(14,116,144,0.035) 0 1px, transparent 1px 36px),
    repeating-linear-gradient(90deg, rgba(14,116,144,0.028) 0 1px, transparent 1px 36px),
    linear-gradient(180deg, #f9fcff 0%, var(--bg) 100%) !important;
  color:var(--ink); font-family:var(--mono);
}

/* Header */
.hud-head{
  border:1px solid var(--line-2); border-radius:12px;
  background:var(--panel);
  padding:24px 28px 22px; margin-bottom:18px; position:relative; overflow:hidden;
  text-align:center;
  box-shadow:0 1px 0 rgba(255,255,255,0.7) inset, 0 10px 26px rgba(32,42,51,0.07);
}
.hud-head:before, .hud-head:after{
  content:""; position:absolute; width:16px; height:16px; opacity:.55;
}
.hud-head:before{left:12px;top:12px;border-left:2px solid var(--teal);border-top:2px solid var(--teal);}
.hud-head:after{right:12px;bottom:12px;border-right:2px solid var(--teal);border-bottom:2px solid var(--teal);}
.hud-title{
  font-family:var(--disp); font-weight:700; font-size:30px; letter-spacing:3px;
  color:var(--ink); margin:0; text-transform:uppercase;
}
.hud-title b{color:var(--teal);}
.hud-sub{font-family:var(--mono);color:var(--muted);font-size:12.5px;letter-spacing:1px;margin-top:8px;}
.hud-tags{margin-top:14px;display:flex;gap:8px;flex-wrap:wrap;justify-content:center;}
.hud-tag{
  font-family:var(--mono);font-size:11px;letter-spacing:1px;color:var(--teal);
  border:1px solid rgba(14,116,144,0.3);border-radius:5px;padding:3px 10px;
  background:rgba(14,116,144,0.07);text-transform:uppercase;
}
.hud-tag.amber{color:var(--amber);border-color:rgba(180,83,9,0.3);background:rgba(180,83,9,0.07);}

/* Equal-height workspace columns */
.work-row{align-items:stretch !important; gap:18px !important;}
.term-panel{
  border:1px solid var(--line-2) !important; border-radius:12px !important;
  background:var(--panel) !important;
  box-shadow:0 1px 0 rgba(255,255,255,0.7) inset, 0 8px 24px rgba(32,42,51,0.06);
  padding:18px !important;
  display:flex !important; flex-direction:column !important;
}
.btn-row{margin-top:auto !important;}
.panel-label{
  font-family:var(--disp);font-weight:600;letter-spacing:2px;text-transform:uppercase;
  color:var(--teal);font-size:13px;margin-bottom:10px;display:flex;align-items:center;gap:9px;
}
.panel-label:before{content:"";width:9px;height:9px;background:var(--amber);
  border-radius:2px;display:inline-block;}

/* Gradio controls */
.gradio-container label span{font-family:var(--mono) !important;color:var(--ink-soft) !important;
  letter-spacing:1px;font-size:12px !important;text-transform:uppercase;}
.gradio-container textarea, .gradio-container input[type=text]{
  font-family:var(--mono) !important;background:var(--surface) !important;
  color:var(--ink) !important;border:1px solid var(--line-2) !important;border-radius:8px !important;}
.gradio-container textarea:focus, .gradio-container input[type=text]:focus{
  border-color:var(--teal) !important; box-shadow:0 0 0 3px rgba(14,116,144,0.14) !important;}
.gradio-container .image-container, .gradio-container [data-testid="image"]{
  background:var(--surface) !important; border-radius:8px !important;}

/* Radio group */
fieldset label{border:1px solid var(--line-2) !important;border-radius:8px !important;
  background:var(--surface) !important;}
fieldset label.selected{border-color:var(--teal) !important;background:rgba(14,116,144,0.08) !important;}

/* Buttons */
button.primary, .gradio-container button.lg.primary{
  font-family:var(--disp) !important;font-weight:600 !important;letter-spacing:2px !important;
  text-transform:uppercase !important;color:#ffffff !important;
  background:linear-gradient(135deg,var(--teal),#0aa2b8) !important;border:none !important;
  box-shadow:0 6px 16px rgba(14,116,144,0.28) !important;
}
button.primary:hover{box-shadow:0 8px 22px rgba(14,116,144,0.42) !important;transform:translateY(-1px);}
button.secondary{
  font-family:var(--mono) !important;letter-spacing:1px !important;text-transform:uppercase !important;
  color:var(--ink-soft) !important;background:var(--surface) !important;
  border:1px solid var(--line-2) !important;
}
button.secondary:hover{color:var(--teal) !important;border-color:var(--teal) !important;}

/* Result readout */
.readout{font-family:var(--mono);font-size:12.5px;line-height:1.5;flex:1;}
.rd-head{font-family:var(--disp);font-weight:600;letter-spacing:1.5px;font-size:14px;
  color:var(--teal);margin-bottom:12px;display:flex;align-items:center;gap:9px;}
.dot{width:9px;height:9px;border-radius:50%;background:var(--teal);
  box-shadow:0 0 0 3px rgba(14,116,144,0.18);display:inline-block;animation:pulse 1.8s infinite;}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}
.rd-row{display:flex;justify-content:space-between;gap:14px;padding:7px 0;
  border-bottom:1px dashed var(--line);}
.rd-key{color:var(--muted);text-transform:uppercase;font-size:11px;letter-spacing:1px;white-space:nowrap;}
.rd-val{color:var(--ink);text-align:right;word-break:break-all;}
.rd-val.accent{color:var(--amber);font-weight:600;}
.rd-raw{margin-top:12px;}
.rd-raw pre{background:var(--surface);border:1px solid var(--line-2);border-left:3px solid var(--teal);
  border-radius:8px;padding:11px 13px;margin-top:6px;color:#0b5566;font-size:11.5px;
  white-space:pre-wrap;word-break:break-word;overflow-x:auto;}
.rd-note{margin-top:14px;padding:11px 13px;border-radius:8px;color:var(--ink);font-size:12px;
  background:rgba(180,83,9,0.07);border:1px solid rgba(180,83,9,0.22);line-height:1.6;}
.rd-note-tag{display:inline-block;color:var(--amber);font-weight:600;letter-spacing:1px;
  margin-right:8px;text-transform:uppercase;font-size:11px;}
.rd-hint{color:var(--muted);font-size:12.5px;line-height:1.7;}

/* Examples */
.ex-title{font-family:var(--disp);font-weight:600;letter-spacing:2px;text-transform:uppercase;
  color:var(--teal);font-size:14px;margin:18px 0 6px;text-align:center;}

.gradio-container table{background:var(--panel) !important;border:1px solid var(--line-2) !important;
  border-radius:10px !important;overflow:hidden;}
.gradio-container thead th, .gradio-container .tr-head th{
  background:rgba(14,116,144,0.08) !important;color:var(--teal) !important;
  font-family:var(--mono) !important;text-transform:uppercase;letter-spacing:1px;
  font-size:11px !important;border-bottom:1px solid var(--line-2) !important;padding:10px 12px !important;}
.gradio-container .tr-body, .gradio-container tbody tr{
  background:var(--panel) !important;
  border-bottom:1px solid var(--line) !important;}
.gradio-container .tr-body:hover, .gradio-container tbody tr:hover{
  background:rgba(14,116,144,0.06) !important;}
.gradio-container td{color:var(--ink) !important;font-family:var(--mono) !important;
  font-size:12.5px !important;border-color:var(--line) !important;}
.gradio-container .paginate, .gradio-container .paginate *{color:var(--muted) !important;}
.gradio-container [class*=dataset]{background:transparent !important;}


/* Hide example-table menus without affecting the result ImagePreview. */
#ex-holder button.svelte-zxsjoa,
.gradio-container .show-api,
.gradio-container .settings,
.gradio-container .record{display:none !important;}

footer{display:none !important;}
.foot{color:var(--muted);font-family:var(--mono);font-size:11px;letter-spacing:.5px;
  text-align:center;margin-top:20px;line-height:1.9;}
.foot .sw{display:inline-block;width:11px;height:11px;border-radius:2px;
  vertical-align:-1px;margin:0 5px 0 14px;border:1px solid rgba(0,0,0,0.2);}
"""

HEADER = """
<div class="hud-head">
  <div class="hud-title">GEO<b>BOX</b>-R1 · VISUAL GROUNDING · 遥感目标定位</div>
  <div class="hud-sub">Describe a target in natural language and get a horizontal or oriented box · 自然语言描述目标，输出水平框 / 旋转框</div>
  <div class="hud-tags">
    <span class="hud-tag">DIOR-RSVG</span><span class="hud-tag">RSVG</span>
    <span class="hud-tag">GeoChat</span><span class="hud-tag">VRSBench</span>
    <span class="hud-tag">AVVG</span>
    <span class="hud-tag amber">HBB · 水平框</span><span class="hud-tag amber">OBB · 旋转框</span>
  </div>
</div>
"""

FOOTER = """
<div class="foot">
  GeoBox-R1 visual grounding demo · 遥感目标定位演示　|
  <span class="sw" style="background:#16B25A"></span>GT · 真值 (dashed · 虚线)
  <span class="sw" style="background:#00B8D9"></span>HBB · 水平框
  <span class="sw" style="background:#F08A08"></span>OBB · 旋转框
  　| greedy decoding, norm1000 coordinates · 贪心解码，norm1000 坐标
</div>
"""


def build_examples():
    # Show oriented examples first while preserving manifest order within each task.
    ordered = sorted(MANIFEST, key=lambda m: 0 if m["recommended_task"] == "obb" else 1)
    rows = []
    for m in ordered:
        path = os.path.join(EXAMPLES_DIR, m["image_file"])
        label = TASK_HBB if m["recommended_task"] == "hbb" else TASK_OBB
        rows.append([path, m["question"], label])
    return rows


with gr.Blocks(title="GeoBox-R1 Demo · 遥感目标定位演示", theme=gr.themes.Base(), css=CSS) as demo:
    gr.HTML(HEADER)

    with gr.Row(equal_height=True, elem_classes="work-row"):
        # Input controls.
        with gr.Column(scale=5, elem_classes="term-panel"):
            gr.HTML('<div class="panel-label">IMAGE INPUT · 影像输入</div>')
            image_in = gr.Image(type="filepath", label="Aerial image · 遥感影像", height=360, sources=["upload"])
            question_in = gr.Textbox(
                label="Query · 目标描述",
                placeholder="e.g. The large baseball field / the car farthest from the camera",
                lines=2,
            )
            task_in = gr.Radio(
                choices=[TASK_HBB, TASK_OBB], value=TASK_HBB,
                label="Task · 定位任务",
            )
            with gr.Row(elem_classes="btn-row"):
                clear_btn = gr.Button("RESET · 清除", variant="secondary", scale=1)
                run_btn = gr.Button("▶ LOCATE · 执行定位", variant="primary", scale=2)

        # Detection result.
        with gr.Column(scale=7, elem_classes="term-panel"):
            gr.HTML('<div class="panel-label">RESULT · 定位结果</div>')
            initial_img, initial_panel = _blank_result("AWAITING INPUT")
            image_out = gr.Image(
                value=initial_img, height=360, interactive=False,
                show_label=False, show_fullscreen_button=True,
                show_download_button=True,
            )
            panel_out = gr.HTML(initial_panel)

    gr.HTML('<div class="ex-title">Examples · 测试样例 — 3 OBB + 2 HBB, click to load · 点击载入</div>')
    examples = gr.Examples(
        examples=build_examples(),
        inputs=[image_in, question_in, task_in],
        examples_per_page=5,
        label="",
        elem_id="ex-holder",
    )

    gr.HTML(FOOTER)

    run_btn.click(run_inference, [image_in, question_in, task_in], [image_out, panel_out])
    # Direct uploads reliably trigger this event.
    image_in.change(on_example_select, [image_in], [question_in, task_in, image_out, panel_out])
    # Example loading bypasses image_in.change, so clear stale output explicitly.
    examples.load_input_event.then(
        lambda: _blank_result("AWAITING INPUT"),
        None,
        [image_out, panel_out],
        queue=False,
    )

    def _reset():
        blank_img, blank_panel = _blank_result("AWAITING INPUT")
        return (gr.update(value=None), gr.update(value=""), gr.update(value=TASK_HBB), blank_img, blank_panel)
    clear_btn.click(_reset, None, [image_in, question_in, task_in, image_out, panel_out])


def _preload():
    try:
        model.load()
    except Exception as e:
        print(f"[GeoBox-R1] Preload failed: {e}")


if __name__ == "__main__":
    # Warm the model while the user opens the interface.
    threading.Thread(target=_preload, daemon=True).start()
    port = int(os.environ.get("GEOBOX_PORT", "7860"))
    demo.queue(max_size=16).launch(
        server_name="0.0.0.0",
        server_port=port,
        share=os.environ.get("GEOBOX_SHARE", "0") == "1",
        show_error=True,
        allowed_paths=[EXAMPLES_DIR],
    )
