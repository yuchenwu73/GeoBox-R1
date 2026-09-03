"""Tests for the baseline evaluation scripts in baselines/evaluate.

The expected values pin the parsing, prompt and scoring behavior that produced the paper's
baseline rows. The scripts must stay importable without their model runtimes (geochat,
llava, lhrs, swift, torch); only the pure functions are exercised here.
"""
import contextlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
EVALUATE_DIR = REPO_ROOT / "baselines" / "evaluate"
if str(EVALUATE_DIR) not in sys.path:
    sys.path.insert(0, str(EVALUATE_DIR))  # the scripts import their shared `common` module by name


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"baseline_{name}", EVALUATE_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


class _Area:
    def __init__(self, area):
        self.area = area


class _RectPolygon:
    """Rectangle-only stand-in for shapely.geometry.Polygon."""

    def __init__(self, points):
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        self.bounds = min(xs), min(ys), max(xs), max(ys)
        self.area = max(0.0, self.bounds[2] - self.bounds[0]) * max(0.0, self.bounds[3] - self.bounds[1])
        self.is_valid = len(points) >= 3 and self.area > 0

    def intersection(self, other):
        left, top = max(self.bounds[0], other.bounds[0]), max(self.bounds[1], other.bounds[1])
        right, bottom = min(self.bounds[2], other.bounds[2]), min(self.bounds[3], other.bounds[3])
        return _Area(max(0.0, right - left) * max(0.0, bottom - top))

    def union(self, other):
        return _Area(self.area + other.area - self.intersection(other).area)


def fake_shapely():
    geometry = types.ModuleType("shapely.geometry")
    geometry.Polygon = _RectPolygon
    shapely = types.ModuleType("shapely")
    shapely.geometry = geometry
    return {"shapely": shapely, "shapely.geometry": geometry}


class CommonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.common = load_script("common")

    def test_dataset_registry(self):
        self.assertEqual(len(self.common.DATASETS), 7)
        self.assertEqual(self.common.OBB_DATASETS, ("geochat_test", "vrsbench_test", "avvg_test"))
        self.assertEqual(self.common.select_datasets("hbb", "all"), list(self.common.DATASETS))
        self.assertEqual(self.common.select_datasets("obb", "all"), list(self.common.OBB_DATASETS))
        self.assertEqual(self.common.select_datasets("obb", "avvg_test"), ["avvg_test"])

    def test_load_jsonl_joins_wrapped_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "meta.jsonl"
            path.write_text('{"a": 1}\n\n{"b":\n[1, 2]}\n{"c": 3}\n', encoding="utf-8")
            self.assertEqual(self.common.load_jsonl(str(path)), [{"a": 1}, {"b": [1, 2]}, {"c": 3}])

    def test_find_image_falls_back_to_other_extensions_and_glob_roots(self):
        with tempfile.TemporaryDirectory() as tmp:
            sub = Path(tmp) / "refGeo_images" / "AVVG"
            sub.mkdir(parents=True)
            Image.new("RGB", (4, 4)).save(sub / "a.png")
            self.assertEqual(self.common.find_image(str(Path(tmp) / "refGeo*"), "AVVG", "a.jpg"), str(sub / "a.png"))
            self.assertIsNone(self.common.find_image(str(Path(tmp) / "refGeo*"), "AVVG", "b.jpg"))

    def test_clip_hbb_orders_and_clips(self):
        self.assertEqual(self.common.clip_hbb([30, 40, 10, 20], (100, 100)), [10.0, 20.0, 30.0, 40.0])
        self.assertEqual(self.common.clip_hbb([-4, 6, 960, 540], (800, 600)), [0.0, 6.0, 799.0, 540.0])

    def test_iou_and_riou(self):
        self.assertAlmostEqual(self.common.iou([0, 0, 10, 10], [5, 0, 15, 10]), 1 / 3)
        self.assertEqual(self.common.iou([0, 0, 0, 0], [0, 0, 0, 0]), 0.0)
        square = [[0, 0], [10, 0], [10, 10], [0, 10]]
        shuffled = [[15, 10], [5, 10], [5, 0], [15, 0]]  # unordered corners must still score
        with mock.patch.dict(sys.modules, fake_shapely()):
            self.assertAlmostEqual(self.common.riou(square, shuffled), 1 / 3)

    @unittest.skipUnless(has_module("shapely"), "shapely not installed")
    def test_riou_with_real_shapely_matches_axis_aligned_iou(self):
        square = [[0, 0], [10, 0], [10, 10], [0, 10]]
        shuffled = [[15, 10], [5, 10], [5, 0], [15, 0]]
        self.assertAlmostEqual(self.common.riou(square, shuffled), 1 / 3)

    def test_summarize_metrics(self):
        self.assertEqual(self.common.summarize([0.9, 0.6, 0.0, 0.5], "hbb"),
                         {"acc@0.5": 0.75, "acc@0.7": 0.25, "mIoU": 0.5})
        self.assertEqual(self.common.summarize([], "obb"), {"acc@0.5": 0.0, "acc@0.7": 0.0, "mRIoU": 0.0})

    def test_run_evaluation_end_to_end(self):
        """Missing images, failing batches and unparseable outputs all score 0 and stay counted."""
        common = self.common
        with tempfile.TemporaryDirectory() as tmp:
            meta = Path(tmp) / "metainfo"
            meta.mkdir()
            images = Path(tmp) / "images" / "AVVG"
            images.mkdir(parents=True)
            for name in ("a.jpg", "b.jpg", "c.jpg"):
                Image.new("RGB", (100, 50)).save(images / name)
            rows = [
                {"question_id": 0, "image_id": "a.jpg", "question": "a", "bbox": [0, 0, 50, 50]},
                {"question_id": 1, "image_id": "b.jpg", "question": "b", "bbox": [0, 0, 50, 50]},
                {"question_id": 2, "image_id": "missing.jpg", "question": "m", "bbox": [0, 0, 50, 50]},
                {"question_id": 3, "image_id": "c.jpg", "question": "c", "bbox": [0, 0, 50, 50]},
                {"question_id": 4, "image_id": "c.jpg", "question": "no gt"},
            ]
            (meta / "avvg_test.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

            calls = []

            def infer_batch(paths, prompts):
                calls.append(list(paths))
                if len(paths) > 1:
                    raise RuntimeError("simulated OOM")  # forces the per-sample retry
                name = os.path.basename(paths[0])
                if name == "b.jpg":
                    raise RuntimeError("bad sample")
                return ["garbage" if name == "c.jpg" else f"{prompts[0]}: 0 0 50 50"]

            def parse(text, size):
                values = [float(v) for v in text.split(":")[-1].split()] if ":" in text else []
                return values if len(values) == 4 else None

            args = types.SimpleNamespace(metainfo_dir=str(meta), image_dir=str(Path(tmp) / "images"),
                                         output_dir=str(Path(tmp) / "out"), batch_size=8, max_samples=None)
            with mock.patch.dict(sys.modules, fake_shapely()), contextlib.redirect_stdout(io.StringIO()):
                summary = common.run_evaluation("hbb", ["avvg_test"], infer_batch, lambda q: f"Q[{q}]", parse,
                                                args, "FakeModel", summary_extra={"note": "x"})

            self.assertEqual(calls[0], [str(images / n) for n in ("a.jpg", "b.jpg", "c.jpg")])
            self.assertEqual(summary["avvg_test"]["count"], 4)
            self.assertEqual(summary["avvg_test"]["infer_errors"], 2)  # failed sample + missing image
            self.assertEqual(summary["avvg_test"]["parse_fail"], 3)
            self.assertAlmostEqual(summary["avvg_test"]["acc@0.5"], 0.25)
            self.assertAlmostEqual(summary["avvg_test"]["mIoU"], 0.25)

            # The run directory sits next to the appended table_*.md file; directory order is not stable.
            run_dir = next(p for p in (Path(tmp) / "out").iterdir() if p.is_dir())
            records = [json.loads(line) for line in (run_dir / "avvg_test_hbb_predictions.jsonl").read_text().splitlines()]
            self.assertEqual([r["question_id"] for r in records], [0, 1, 2, 3])
            self.assertEqual(records[0]["output"], "Q[a]: 0 0 50 50")
            self.assertEqual(records[0]["prediction"], [0.0, 0.0, 50.0, 50.0])
            self.assertTrue(records[1]["error"].startswith("ERROR: bad sample"))
            self.assertEqual(records[1]["output"], "")
            self.assertEqual(records[2]["error"], "image not found")
            self.assertIsNone(records[3]["prediction"])
            written = json.loads((run_dir / "summary_hbb.json").read_text())
            self.assertEqual(written["model"], "FakeModel")
            self.assertEqual(written["note"], "x")
            table = (run_dir / "evaluation_metrics.md").read_text()
            self.assertIn('<th colspan="3">AVG</th>', table)
            self.assertIn("25.00", table)
            self.assertTrue((Path(tmp) / "out" / "table_hbb_acc05_acc07_miou.md").exists())


class GeoChatTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = load_script("eval_geochat")

    def test_hbb_uses_the_envelope_of_the_rotated_box(self):
        self.assertEqual(self.m.hbb("{<76><59><80><67>|<45>}", (1000, 800)),
                         [743.2304473782996, 467.2304473782995, 816.7695526217004, 540.7695526217004])
        self.assertEqual(self.m.hbb("{<10><20><30><40>|<0>}", (1000, 800)), [100.0, 160.0, 300.0, 320.0])
        self.assertEqual(self.m.hbb("{<10><20><30><40>}", (1000, 800)), [100.0, 160.0, 300.0, 320.0])
        self.assertIsNone(self.m.hbb("no coordinates here", (100, 100)))

    def test_obb_requires_the_angle(self):
        self.assertEqual(self.m.obb("{<76><59><80><67>|<45>}", (1000, 800)),
                         [[788.4852813742385, 467.2304473782995], [816.7695526217004, 495.5147186257614],
                          [771.5147186257615, 540.7695526217004], [743.2304473782996, 512.4852813742385]])
        self.assertEqual(self.m.obb("{<10><20><30><40>|<0>}", (1000, 800)),
                         [[100.0, 160.0], [300.0, 160.0], [300.0, 320.0], [100.0, 320.0]])
        self.assertIsNone(self.m.obb("{<10><20><30><40>}", (1000, 800)))

    def test_prompt_is_the_official_referring_template(self):
        self.assertEqual(self.m.task_prompt("hbb", "the red car"), "[refer] Give me the location of <p> the red car </p>")
        self.assertEqual(self.m.task_prompt("obb", "the red car"), self.m.task_prompt("hbb", "the red car"))


class GeoGroundTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = load_script("eval_geoground")

    def test_prompts(self):
        self.assertEqual(self.m.get_hbb_prompt("the red car"),
                         "[refer] output the bounding box of the <ref>the red car</ref> in the image.")
        self.assertEqual(self.m.get_obb_prompt("the red car"),
                         "[refer] output the oriented bounding box of the <ref>the red car</ref> in the image.")

    def test_hbb_norm1000_to_clipped_pixels(self):
        self.assertEqual(self.m.parse_hbb_response("<box>[[80, 578, 232, 725]]</box>", (4000, 2250)),
                         [320.0, 1300.5, 928.0, 1631.25])
        self.assertEqual(self.m.parse_hbb_response("<box>[[-5, 10, 1200, 900]]</box>", (800, 600)),
                         [0.0, 6.0, 799.0, 540.0])
        self.assertIsNone(self.m.parse_hbb_response("I cannot find it", (800, 600)))

    @unittest.skipUnless(has_module("cv2"), "opencv not installed")
    def test_obb_le90_uses_one_length_scale(self):
        self.assertEqual(self.m.parse_obb_response("<obb>[[15, 65, 15, 14, 0]]</obb>", (4000, 2250)),
                         [[300.0, 1742.5], [300.0, 1182.5], [900.0, 1182.5], [900.0, 1742.5]])
        self.assertEqual(self.m.parse_obb_response("<obb>[[50, 50, 20, 10, -90]]</obb>", (1000, 500)),
                         [[550.0, 350.0], [450.0, 350.0], [450.0, 150.0], [550.0, 150.0]])

    def test_obb_fallbacks(self):
        self.assertEqual(self.m.parse_obb_response("[[10, 20, 30, 20, 30, 40, 10, 40]]", (1000, 500)),
                         [[10.0, 20.0], [30.0, 20.0], [30.0, 40.0], [10.0, 40.0]])
        self.assertIsNone(self.m.parse_obb_response("<box>[[10, 20, 30, 40]]</box>", (1000, 500)))


class InternVLTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = load_script("eval_internvl")

    def test_prompts(self):
        self.assertEqual(
            self.m.prompt("hbb", "the red car"),
            "Please provide the bounding box coordinate of the region this sentence describes: <ref>the red car</ref>"
            " Answer with only the bounding box in the format <box>[[x1,y1,x2,y2]]</box>.")
        self.assertEqual(
            self.m.prompt("obb", "the red car"),
            "Locate the region described by: <ref>the red car</ref>. Return only its oriented bounding box as four "
            "clockwise corner points normalized to [0,1000]: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]].")

    def test_hbb(self):
        self.assertEqual(self.m.parse_hbb("<ref>The gray court</ref><box>[[766, 543, 1000, 810]]</box>", (800, 800)),
                         [612.8, 434.40000000000003, 799.0, 648.0])
        self.assertIsNone(self.m.parse_hbb("nothing", (800, 800)))

    def test_obb_is_strict(self):
        self.assertIsNone(self.m.parse_obb("<box>[[766, 543, 1000, 810]]</box>", (800, 800)))
        self.assertEqual(self.m.parse_obb("[[100,100],[300,100],[300,200],[100,200]]", (1000, 500)),
                         [[100.0, 50.0], [300.0, 50.0], [300.0, 100.0], [100.0, 100.0]])
        self.assertIsNone(self.m.parse_obb("[[100,100,300,200],[100,100,300,200]]", (1000, 500)))
        self.assertEqual(self.m.parse_obb("[100, 100, 300, 200, 0]", (1000, 500)),
                         [[100.0, 50.0], [300.0, 50.0], [300.0, 100.0], [100.0, 100.0]])

    def test_response_text_only_takes_the_generated_content(self):
        message = types.SimpleNamespace(content="<box>x</box>")
        response = types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)], usage={"tokens": 123})
        self.assertEqual(self.m.response_text(response), "<box>x</box>")
        self.assertEqual(self.m.response_text({"choices": [{"message": {"content": "y"}}]}), "y")
        self.assertEqual(self.m.response_text("z"), "z")


class LHRSTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = load_script("eval_lhrs")

    def test_specs(self):
        self.assertEqual(self.m.MODEL_SPECS["lhrs"]["prefix"],
                         "[VG] Please output the coordinate of the following object: ")
        self.assertEqual(self.m.MODEL_SPECS["lhrs-nova"]["prefix"],
                         "[DET] Please output the coordinate of the following object: ")
        for spec in self.m.MODEL_SPECS.values():
            for key in ("repo", "ckpt", "text_path", "vit_name"):
                self.assertTrue(spec[key].startswith("models/pretrained/"), spec[key])

    def test_hbb(self):
        self.assertEqual(self.m.parse_hbb("[0.17,0.3,0.86,0.84]", (800, 600), "lhrs"), [136.0, 180.0, 688.0, 504.0])
        nova = "The object is at <bbox>[0.1, 0.2, 0.5, 0.6]</bbox>."
        self.assertEqual(self.m.parse_hbb(nova, (800, 600), "lhrs-nova"), [80.0, 120.0, 400.0, 360.0])
        self.assertEqual(self.m.parse_hbb("[0.1, 0.2, 0.5, 0.6, 0.9]", (800, 600), "lhrs"), [80.0, 120.0, 400.0, 360.0])
        self.assertIsNone(self.m.parse_hbb("no box", (800, 600), "lhrs"))


class HFBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = load_script("eval_hf")

    def test_trained_prompts_match_the_main_evaluation(self):
        self.assertEqual(self.m.build_prompt("the red car", "hbb", "trained", "qwenvl"),
                         self.m.hbb_base.get_prompt("the red car"))
        self.assertEqual(self.m.build_prompt("the red car", "obb", "trained", "qwenvl"),
                         self.m.obb_base.get_obb_prompt("the red car"))

    def test_zeroshot_prompts(self):
        self.assertEqual(
            self.m.build_prompt("the red car", "hbb", "zeroshot", "qwenvl"),
            'Locate the object this description refers to: "the red car". Output its bounding box in JSON format: '
            '{"bbox_2d": [x1, y1, x2, y2]} using absolute pixel coordinates.')
        self.assertEqual(self.m.build_prompt("the red car", "obb", "zeroshot", "qwenvl"),
                         self.m.obb_base.get_obb_prompt("the red car") + "\nUse absolute pixel coordinates.")
        self.assertEqual(self.m.build_prompt("the red car", "hbb", "zeroshot", "generic"),
                         self.m.hbb_base.get_prompt("the red car"))

    def test_lenient_parsers(self):
        self.assertEqual(self.m.parse_hbb_tolerant('```json\n[{"horizontal_bbox": [100, 200, 700, 800]}]\n```'),
                         [100.0, 200.0, 700.0, 800.0])
        self.assertEqual(self.m.parse_hbb_tolerant('{"bbox_2d": [10, 20, 30, 40]}'), [10.0, 20.0, 30.0, 40.0])
        self.assertEqual(self.m.parse_hbb_tolerant("<box>(10,20),(30,40)</box>"), [10.0, 20.0, 30.0, 40.0])
        self.assertEqual(self.m.parse_hbb_tolerant("x 1 2 3 4 5"), [1.0, 2.0, 3.0, 4.0])
        self.assertIsNone(self.m.parse_hbb_tolerant("nothing 1 2"))
        self.assertEqual(self.m.parse_obb_tolerant('[{"oriented_bbox": [[10, 20], [30, 20], [30, 40], [10, 40]]}]'),
                         [[10.0, 20.0], [30.0, 20.0], [30.0, 40.0], [10.0, 40.0]])
        self.assertEqual(self.m.parse_obb_tolerant("1 2 3 4 5 6 7 8 9"),
                         [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        self.assertIsNone(self.m.parse_obb_tolerant("1 2 3"))

    def test_coordinate_spaces(self):
        self.assertEqual(self.m.scale_coords([0.5, 0.25, 1.0, 1.0], "norm1", 800, 400), [400.0, 100.0, 800.0, 400.0])
        self.assertEqual(self.m.scale_coords([500, 250, 1000, 1000], "norm1000", 800, 400), [400.0, 100.0, 800.0, 400.0])
        self.assertEqual(self.m.scale_coords([5, 6, 7, 8], "abs", 800, 400), [5, 6, 7, 8])
        self.assertEqual(self.m.auto_space([0.1, 0.2, 0.9, 1.0], "abs", "auto"), "norm1")
        self.assertEqual(self.m.auto_space([10, 20, 900, 1000], "abs", "auto"), "abs")
        self.assertEqual(self.m.auto_space([0.1, 0.2, 0.9, 1.0], "abs", "abs"), "abs")
        resolve = self.m.resolve_coord_space
        self.assertEqual([resolve("auto", "qwenvl"), resolve("auto", "generic"), resolve("norm100", "qwenvl")],
                         ["abs", "norm1000", "norm100"])

    def test_to_pixels(self):
        trained = types.SimpleNamespace(prompt_mode="trained", coord_space="auto")
        self.assertEqual(self.m.to_pixels([100, 200, 700, 800], "hbb", trained, 2000, 1000, "norm1000"),
                         [200.0, 200.0, 1400.0, 800.0])
        corners = [[100, 200], [700, 200], [700, 800], [100, 800]]
        self.assertEqual(self.m.to_pixels(corners, "obb", trained, 2000, 1000, "norm1000"),
                         [[200.0, 200.0], [1400.0, 200.0], [1400.0, 800.0], [200.0, 800.0]])
        norm1 = types.SimpleNamespace(prompt_mode="trained", coord_space="norm1")
        self.assertEqual(self.m.to_pixels([0.5, 0.5, 1.0, 1.0], "hbb", norm1, 800, 400, "norm1000"),
                         [400.0, 200.0, 800.0, 400.0])
        zeroshot = types.SimpleNamespace(prompt_mode="zeroshot", coord_space="auto")
        self.assertEqual(self.m.to_pixels([0.5, 0.5, 1.0, 1.0], "hbb", zeroshot, 800, 400, "abs"),
                         [400.0, 200.0, 800.0, 400.0])
        self.assertEqual(self.m.to_pixels([400, 200, 800, 400], "hbb", zeroshot, 800, 400, "abs"), [400, 200, 800, 400])


class ScriptHygieneTests(unittest.TestCase):
    def test_no_cjk_and_no_stale_references(self):
        import re

        cjk = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
        for path in sorted(EVALUATE_DIR.glob("*.py")):
            text = path.read_text(encoding="utf-8")
            self.assertIsNone(cjk.search(text), path)
            for stale in ("_A100", "rebuttal", "EVAL_PROMPT_SUFFIX", "baseline_finetune"):
                self.assertNotIn(stale, text, f"{path.name} still mentions {stale}")

    def test_scripts_expose_a_main_and_usage(self):
        for name in ("eval_geochat", "eval_geoground", "eval_internvl", "eval_lhrs", "eval_hf"):
            module = load_script(name)
            self.assertTrue(callable(module.main), name)
            self.assertIn("baselines/evaluate/" + name + ".py", module.__doc__, name)


if __name__ == "__main__":
    unittest.main()
