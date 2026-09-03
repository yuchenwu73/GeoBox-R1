"""Tests for demo/: the inference wrapper, example preparation and the bundled assets."""

import json
import os
import shutil
import sys
import tempfile
import threading
import time
import types
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

from _support import DEMO_DIR, REPO_ROOT, load_module


class DemoRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.demo = load_module("demo/inference.py")

    def test_infer_rejects_unknown_task_before_loading_the_model(self):
        model = self.demo.GeoBoxR1()
        model.load = mock.Mock()
        model._engine = mock.Mock()
        model._engine.infer.return_value = [
            {
                "choices": [
                    {
                        "message": {
                            "content": '[{"oriented_bbox": [[0, 0], [1, 0], [1, 1], [0, 1]]}]'
                        }
                    }
                ]
            }
        ]
        with self.assertRaises(ValueError):
            model.infer(
                str(DEMO_DIR / "examples" / "01_dior_rsvg_09017.jpg"),
                "target",
                "polygon",
            )
        model.load.assert_not_called()

    def test_infer_removes_pil_temporary_file_after_success(self):
        from PIL import Image

        captured_paths = []

        class FakeEngine:
            def infer(self, requests, **kwargs):
                captured_paths.append(requests[0]["images"][0])
                return [
                    {
                        "choices": [
                            {
                                "message": {
                                    "content": '[{"horizontal_bbox": [0, 0, 1000, 1000]}]'
                                }
                            }
                        ]
                    }
                ]

        model = self.demo.GeoBoxR1()
        model._engine = FakeEngine()
        model._request_config = object()
        model.load = mock.Mock()
        result = model.infer(Image.new("RGB", (12, 8)), "target", "hbb")
        self.assertTrue(result["parsed_ok"])
        self.assertEqual(len(captured_paths), 1)
        leaked = Path(captured_paths[0]).exists()
        Path(captured_paths[0]).unlink(missing_ok=True)
        self.assertFalse(leaked)

    def test_infer_removes_pil_temporary_file_after_engine_error(self):
        from PIL import Image

        captured_paths = []

        class FailingEngine:
            def infer(self, requests, **kwargs):
                captured_paths.append(requests[0]["images"][0])
                raise RuntimeError("inference failed")

        model = self.demo.GeoBoxR1()
        model._engine = FailingEngine()
        model._request_config = object()
        model.load = mock.Mock()
        with self.assertRaisesRegex(RuntimeError, "inference failed"):
            model.infer(Image.new("RGB", (12, 8)), "target", "hbb")
        self.assertEqual(len(captured_paths), 1)
        leaked = Path(captured_paths[0]).exists()
        Path(captured_paths[0]).unlink(missing_ok=True)
        self.assertFalse(leaked)

    def test_concurrent_load_initializes_the_engine_once(self):
        initialization_count = 0
        count_lock = threading.Lock()

        class FakeEngine:
            def __init__(self, **kwargs):
                nonlocal initialization_count
                with count_lock:
                    initialization_count += 1
                time.sleep(0.05)

        torch_module = types.ModuleType("torch")
        torch_module.bfloat16 = object()
        infer_engine_module = types.ModuleType("swift.infer_engine")
        infer_engine_module.TransformersEngine = FakeEngine
        protocol_module = types.ModuleType("swift.infer_engine.protocol")
        protocol_module.RequestConfig = lambda **kwargs: kwargs
        swift_module = types.ModuleType("swift")
        swift_module.infer_engine = infer_engine_module
        model = self.demo.GeoBoxR1()
        modules = {
            "torch": torch_module,
            "swift": swift_module,
            "swift.infer_engine": infer_engine_module,
            "swift.infer_engine.protocol": protocol_module,
        }
        with mock.patch.dict(sys.modules, modules):
            with ThreadPoolExecutor(max_workers=4) as executor:
                list(executor.map(lambda _: model.load(), range(4)))
        self.assertEqual(initialization_count, 1)

    def test_prepare_examples_default_data_root_is_repository_anchored(self):
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.dict(
            os.environ, {}, clear=False
        ):
            os.environ.pop("REFGEO_ROOT", None)
            try:
                os.chdir(temp_dir)
                module = load_module("demo/prepare_examples.py")
                resolved_root = Path(module.REFGEO_ROOT).resolve()
            finally:
                os.chdir(original_cwd)
        self.assertEqual(resolved_root, REPO_ROOT / "data" / "refGeo")

    def test_prepare_examples_does_not_replace_manifest_when_a_sample_is_missing(self):
        module = load_module("demo/prepare_examples.py")
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            meta = root / "metainfo"
            meta.mkdir()
            (meta / "missing_test.jsonl").write_text("", encoding="utf-8")
            manifest = root / "examples_manifest.json"
            manifest.write_text('{"preserve": true}', encoding="utf-8")
            module.META = str(meta)
            module.IMG_BASE = str(root / "images")
            module.EXAMPLES_DIR = str(root / "examples")
            module.MANIFEST = str(manifest)
            module.SUBDIR = {"missing_test": "missing"}
            module.DATASET_LABEL = {"missing_test": "Missing"}
            module.SELECTED = [("missing_test", "none.png", "target", "hbb", "note")]
            with self.assertRaises(RuntimeError):
                module.main()
            self.assertEqual(manifest.read_text(encoding="utf-8"), '{"preserve": true}')

    def test_prepare_examples_stages_images_before_replacing_published_assets(self):
        from PIL import Image

        module = load_module("demo/prepare_examples.py")
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            examples = root / "examples"
            examples.mkdir()
            old_image = examples / "old.png"
            old_image.write_bytes(b"original")
            manifest = root / "examples_manifest.json"
            manifest.write_text('{"preserve": true}', encoding="utf-8")
            sources = []
            records = []
            selected = []
            for index in (1, 2):
                source = root / f"source-{index}.png"
                Image.new("RGB", (8, 8), (index, index, index)).save(source)
                sources.append(str(source))
                image_id = f"image-{index}.png"
                question = f"target {index}"
                records.append({"image_id": image_id, "question": question, "bbox": [0, 0, 1, 1]})
                selected.append(("sample_test", image_id, question, "hbb", "note"))
            module.META = str(root / "metainfo")
            module.IMG_BASE = str(root / "images")
            module.EXAMPLES_DIR = str(examples)
            module.MANIFEST = str(manifest)
            module.SUBDIR = {"sample_test": "sample"}
            module.DATASET_LABEL = {"sample_test": "Sample"}
            module.SELECTED = selected
            real_copy = shutil.copy2
            copy_count = 0

            def fail_second_copy(source, destination):
                nonlocal copy_count
                copy_count += 1
                if copy_count == 2:
                    raise OSError("copy failed")
                return real_copy(source, destination)

            with mock.patch.object(module, "load_jsonl", return_value=records), mock.patch.object(
                module, "resolve_image", side_effect=sources
            ), mock.patch.object(module.shutil, "copy2", side_effect=fail_second_copy):
                with self.assertRaisesRegex(OSError, "copy failed"):
                    module.main()
            self.assertEqual(manifest.read_text(encoding="utf-8"), '{"preserve": true}')
            self.assertEqual({path.name for path in examples.iterdir()}, {"old.png"})
            self.assertEqual(old_image.read_bytes(), b"original")

    def test_manifest_entries_correspond_exactly_to_example_images(self):
        manifest = json.loads((DEMO_DIR / "examples_manifest.json").read_text(encoding="utf-8"))
        manifest_files = {entry["image_file"] for entry in manifest}
        example_files = {path.name for path in (DEMO_DIR / "examples").iterdir() if path.is_file()}
        self.assertEqual(len(manifest), 5)
        self.assertEqual(manifest_files, example_files)

    def test_demo_ui_does_not_default_unknown_task_labels_to_hbb(self):
        source = (DEMO_DIR / "app.py").read_text(encoding="utf-8")
        self.assertNotIn('TASK_MAP.get(task_label, "hbb")', source)

    def test_demo_proxy_dependency_is_declared(self):
        requirements = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")
        self.assertRegex(requirements, r"(?m)^socksio(?:[<>=!~].*)?$")


if __name__ == "__main__":
    unittest.main()
