#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Single-image inference utilities for the GeoBox-R1 demo."""

import os

# These variables must be set before importing Swift or Torch.
os.environ.setdefault("IMAGE_MAX_TOKEN_NUM", "1024")
os.environ.setdefault("QWENVL_BBOX_FORMAT", "new")

import json
import re
import threading
import time
from typing import List, Optional, Tuple, Union

# The model may be a local checkpoint or a Hugging Face repository ID.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL_PATH = os.environ.get(
    "GEOBOX_MODEL", os.path.join(_REPO_ROOT, "models", "checkpoints", "GeoBox-R1")
)


def hbb_prompt(question: str) -> str:
    return (
        f"Locate the instance that matches the description: [{question}]. "
        "Report horizontal bbox coordinates in following JSON format:\n"
        "```json\n[\n\t{\"horizontal_bbox\": [x1, y1, x2, y2]}\n]\n```"
    )


def obb_prompt(question: str) -> str:
    return (
        f"Locate the instance that matches the description: [{question}]. "
        "Report oriented bbox coordinates in following JSON format:\n"
        "```json\n[\n\t{\"oriented_bbox\": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]}\n]\n```"
    )


def parse_hbb(output: str) -> Optional[List[float]]:
    """Extract a schema-conforming HBB without executing generated text."""
    try:
        text = output.replace("```json", "").replace("```", "").strip()
        match = re.search(r"\[[\s\S]*?\{[\s\S]*?\}[\s\S]*?\]", text)
        if match:
            data = json.loads(match.group())
            if isinstance(data, list) and data:
                item = data[0]
                if isinstance(item, dict) and "horizontal_bbox" in item:
                    bbox = item["horizontal_bbox"]
                    if isinstance(bbox, list) and len(bbox) == 4:
                        return [float(x) for x in bbox]
    except Exception:
        pass
    return None


def parse_obb(output: str) -> Optional[List[List[float]]]:
    """Use the last valid OBB because reasoning may contain earlier draft boxes."""
    try:
        text = output.replace("```json", "").replace("```", "").strip()
        for snippet in reversed(re.findall(r"\[[\s\S]*?\{[\s\S]*?\}[\s\S]*?\]", text)):
            try:
                data = json.loads(snippet)
            except Exception:
                continue
            if isinstance(data, list) and data:
                item = data[0]
                if isinstance(item, dict) and "oriented_bbox" in item:
                    poly = item["oriented_bbox"]
                    if isinstance(poly, list) and len(poly) == 4 and isinstance(poly[0], (list, tuple)):
                        return [[float(pt[0]), float(pt[1])] for pt in poly]
    except Exception:
        pass
    return None


def denorm_hbb(bbox: List[float], size: Tuple[int, int]) -> List[float]:
    """Map norm1000 HBB coordinates to original-image pixels."""
    w, h = size
    return [bbox[0] / 1000.0 * w, bbox[1] / 1000.0 * h,
            bbox[2] / 1000.0 * w, bbox[3] / 1000.0 * h]


def denorm_obb(poly: List[List[float]], size: Tuple[int, int]) -> List[List[float]]:
    """Map norm1000 OBB vertices to original-image pixels."""
    w, h = size
    return [[pt[0] / 1000.0 * w, pt[1] / 1000.0 * h] for pt in poly]


def _extract_text(response) -> str:
    """Normalize the response shapes returned by supported inference engines."""
    if isinstance(response, list) and response:
        response = response[0]
    if hasattr(response, "choices"):
        return response.choices[0].message.content
    if isinstance(response, dict):
        return response["choices"][0]["message"]["content"]
    return str(response)


def iou_hbb(a: List[float], b: List[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def iou_obb(p1: List[List[float]], p2: List[List[float]]) -> Optional[float]:
    """Compute polygon IoU after ordering vertices around each centroid."""
    try:
        from shapely.geometry import Polygon

        def order(pts):
            cx = sum(p[0] for p in pts) / len(pts)
            cy = sum(p[1] for p in pts) / len(pts)
            import math
            return sorted(pts, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))

        poly1 = Polygon(order([(float(p[0]), float(p[1])) for p in p1]))
        poly2 = Polygon(order([(float(p[0]), float(p[1])) for p in p2]))
        if not poly1.is_valid or not poly2.is_valid:
            return None
        inter = poly1.intersection(poly2).area
        union = poly1.union(poly2).area
        return inter / union if union > 0 else 0.0
    except Exception:
        return None


class GeoBoxR1:
    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        model_type: str = "qwen3_vl",
        max_new_tokens: int = 512,
        attn_impl: str = os.environ.get("GEOBOX_ATTN", "flash_attn"),  # "sdpa" if flash-attn is absent
    ):
        self.model_path = model_path
        self.model_type = model_type
        self.max_new_tokens = max_new_tokens
        self.attn_impl = attn_impl
        self._engine = None
        self._request_config = None
        self._load_lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._engine is not None

    def load(self):
        """Load the model once, including under concurrent first requests."""
        if self._engine is not None:
            return
        # Preloading and the first UI request can race during startup.
        with self._load_lock:
            if self._engine is not None:
                return
            import torch
            from swift.infer_engine import TransformersEngine
            from swift.infer_engine.protocol import RequestConfig

            print(f"[GeoBox-R1] Loading model: {self.model_path}")
            t0 = time.time()
            engine = TransformersEngine(
                model=self.model_path,
                model_type=self.model_type if self.model_type else None,
                torch_dtype=torch.bfloat16,
                max_batch_size=1,
                attn_impl=self.attn_impl,
            )
            request_config = RequestConfig(max_tokens=self.max_new_tokens, temperature=0.0)
            self._engine = engine
            self._request_config = request_config
            print(f"[GeoBox-R1] Model loaded in {time.time() - t0:.1f}s")

    def _save_if_needed(self, image: Union[str, "object"]) -> str:
        """Return an image path, creating a temporary PNG for PIL images."""
        if isinstance(image, str):
            return image
        import tempfile
        from PIL import Image as PILImage
        if isinstance(image, PILImage.Image):
            fd, path = tempfile.mkstemp(suffix=".png")
            os.close(fd)
            try:
                image.save(path)
            except Exception:
                os.remove(path)
                raise
            return path
        raise TypeError(f"Unsupported image type: {type(image)}")

    def infer(self, image, question: str, task: str = "hbb") -> dict:
        """Run grounding for one image and return normalized and pixel coordinates."""
        task = task.lower()
        if task not in {"hbb", "obb"}:
            raise ValueError(f"Unsupported task: {task!r}; expected 'hbb' or 'obb'")

        self.load()
        image_path = self._save_if_needed(image)
        temporary_path = image_path if not isinstance(image, str) else None
        try:
            from PIL import Image as PILImage
            with PILImage.open(image_path) as im:
                size = im.size

            prompt = hbb_prompt(question) if task == "hbb" else obb_prompt(question)
            request = {
                "messages": [{"role": "user", "content": f"<image>{prompt}"}],
                "images": [image_path],
            }

            t0 = time.time()
            responses = self._engine.infer([request], request_config=self._request_config, use_tqdm=False)
            latency = time.time() - t0
            raw = _extract_text(responses[0] if isinstance(responses, list) else responses)
        finally:
            # Gradio may pass an in-memory PIL image; never retain its temporary copy.
            if temporary_path:
                try:
                    os.remove(temporary_path)
                except FileNotFoundError:
                    pass

        result = {
            "task": task,
            "raw_output": raw,
            "latency": latency,
            "image_size": size,
            "parsed_ok": False,
            "norm_coords": None,
            "bbox_px": None,
            "poly_px": None,
        }

        if task == "hbb":
            norm = parse_hbb(raw)
            if norm is not None:
                result["parsed_ok"] = True
                result["norm_coords"] = norm
                result["bbox_px"] = denorm_hbb(norm, size)
        else:
            norm = parse_obb(raw)
            if norm is not None:
                result["parsed_ok"] = True
                result["norm_coords"] = norm
                result["poly_px"] = denorm_obb(norm, size)

        return result


_SINGLETON: Optional[GeoBoxR1] = None


def get_model(model_path: str = DEFAULT_MODEL_PATH) -> GeoBoxR1:
    global _SINGLETON
    if _SINGLETON is None:
        _SINGLETON = GeoBoxR1(model_path=model_path)
    return _SINGLETON


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    manifest = json.load(open(os.path.join(here, "examples_manifest.json"), encoding="utf-8"))
    sample = manifest[0]
    img = os.path.join(here, "examples", sample["image_file"])
    model = get_model()
    out = model.infer(img, sample["question"], sample["recommended_task"])
    print("Task:", out["task"])
    print("Raw output:", out["raw_output"])
    print("Normalized coordinates:", out["norm_coords"])
    print("Pixel coordinates:", out.get("bbox_px") or out.get("poly_px"))
    print(f"Latency: {out['latency']:.2f}s")
    if out["task"] == "hbb" and out["bbox_px"]:
        print("IoU:", iou_hbb(out["bbox_px"], sample["gt_bbox"]))
