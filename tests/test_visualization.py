"""Tests for visualization/: default paths, the tau analyses, the multi-model comparison
scheduler, and the plotting scripts."""

import ast
import contextlib
import importlib.util
import io
import json
import runpy
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from _support import (
    EVALUATION_DIR,
    REPO_ROOT,
    VISUALIZATION_DIR,
    assert_public_rl_path,
    assigned_value_node,
    load_function_subset,
    load_module,
)

try:
    import cv2  # noqa: F401
    HAS_CV2 = True
except Exception:  # pragma: no cover - depends on the environment
    HAS_CV2 = False


def load_script(path: Path):
    """Import a script by path under a private module name (siblings stay importable)."""
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))  # export_sample.py imports compare_testsets
    name = "vis_test_" + path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_pure_functions(path: Path, names):
    """Execute only stdlib imports, constant assignments and the named functions of a script."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    stdlib = {"os", "sys", "re", "json", "math", "argparse", "pathlib", "typing", "ast"}
    body = []
    for node in tree.body:
        if isinstance(node, ast.Import) and all(alias.name.split(".")[0] in stdlib for alias in node.names):
            body.append(node)
        elif isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] in stdlib:
            body.append(node)
        elif isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            body.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in names:
            body.append(node)
    namespace = {"__file__": str(path)}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)
    return types.SimpleNamespace(**{name: namespace[name] for name in names})


class InferAndPlotTests(unittest.TestCase):
    NAMES = {"hbb_prompt", "obb_prompt", "denorm_hbb", "denorm_obb", "_parse_gpu_early"}

    @classmethod
    def setUpClass(cls):
        cls.plot = load_pure_functions(VISUALIZATION_DIR / "infer_and_plot.py", cls.NAMES)
        cls.hbb = load_pure_functions(EVALUATION_DIR / "evaluate_hbb.py", {"get_prompt"})
        cls.obb = load_pure_functions(EVALUATION_DIR / "evaluate_obb.py", {"get_obb_prompt"})

    def test_prompts_match_the_evaluation_prompts(self):
        question = "the white plane near the terminal"
        self.assertEqual(self.plot.hbb_prompt(question), self.hbb.get_prompt(question))
        self.assertEqual(self.plot.obb_prompt(question), self.obb.get_obb_prompt(question))
        self.assertNotIn("<image>", self.plot.hbb_prompt(question))

    def test_norm1000_boxes_are_mapped_to_original_pixels(self):
        self.assertEqual(self.plot.denorm_hbb([100, 200, 700, 800], (2000, 1000)), [200.0, 200.0, 1400.0, 800.0])
        self.assertEqual(
            self.plot.denorm_obb([[100, 200], [700, 200], [700, 800], [100, 800]], (2000, 1000)),
            [[200.0, 200.0], [1400.0, 200.0], [1400.0, 800.0], [200.0, 800.0]],
        )

    def test_gpu_override_is_only_applied_when_requested(self):
        self.assertIsNone(self.plot._parse_gpu_early(["prog", "--image", "a.png"]))
        self.assertEqual(self.plot._parse_gpu_early(["prog", "--gpu", "3"]), "3")
        self.assertEqual(self.plot._parse_gpu_early(["prog", "--gpu=1"]), "1")

    @unittest.skipUnless(HAS_CV2, "cv2 not installed")
    def test_module_imports_without_touching_cuda_visible_devices(self):
        with mock.patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "5"}), mock.patch.object(sys, "argv", ["prog"]):
            import os
            load_script(VISUALIZATION_DIR / "infer_and_plot.py")
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "5")


class CompareTestsetsTests(unittest.TestCase):
    def test_obb_split_falls_back_to_the_full_test_file(self):
        module = load_script(VISUALIZATION_DIR / "compare_testsets.py")
        with tempfile.TemporaryDirectory() as temp_dir:
            metainfo = Path(temp_dir)
            (metainfo / "geochat_test.jsonl").write_text("{}\n", encoding="utf-8")
            spec = module.DatasetSpec(
                "geochat_test_filtered", "obb", metainfo / "OBB_Selected" / "geochat_test_filtered.jsonl",
                "GeoChat", "GeoChat", 8,
            )
            with mock.patch.object(module, "METAINFO_DIR", metainfo), mock.patch.object(module, "log"):
                self.assertEqual(module.resolve_metainfo_file(spec), metainfo / "geochat_test.jsonl")
            (metainfo / "OBB_Selected").mkdir()
            spec.metainfo_file.write_text("{}\n", encoding="utf-8")
            with mock.patch.object(module, "METAINFO_DIR", metainfo):
                self.assertEqual(module.resolve_metainfo_file(spec), spec.metainfo_file)


class ExportSampleTests(unittest.TestCase):
    def test_select_sample_defaults_to_the_first_sample(self):
        module = load_script(VISUALIZATION_DIR / "export_sample.py")
        samples = [
            {"uid": "a", "question_id": 7, "image_id": "x.png", "image_path": "/tmp/x.png"},
            {"uid": "b", "question_id": 8, "image_id": "y.png", "image_path": "/tmp/y.png"},
        ]
        args = types.SimpleNamespace(uid=None, question_id=None, image_id=None, dataset="d")
        self.assertEqual(module.select_sample(samples, args)["uid"], "a")
        args.question_id = 8
        self.assertEqual(module.select_sample(samples, args)["uid"], "b")
        args.question_id = 9
        with self.assertRaises(FileNotFoundError):
            module.select_sample(samples, args)



class VisualizationConfigurationTests(unittest.TestCase):
    def test_compare_testsets_uses_repository_root_for_default_paths(self):
        module = load_module("visualization/compare_testsets.py")
        self.assertEqual(module.REPO_DIR, REPO_ROOT)
        for path in (
            module.IMAGE_BASE_DIR,
            module.METAINFO_DIR,
            module.DEFAULT_BASE_MODEL,
            module.DEFAULT_SFT_MODEL,
            module.DEFAULT_GDPO_ROOT,
        ):
            self.assertTrue(path.is_relative_to(REPO_ROOT), path)

    def test_gdpo_bundle_loads_the_final_merged_model(self):
        module = load_module("visualization/compare_testsets.py")
        self.assertEqual(
            module.model_bundle("gdpo"),
            {
                "backend": "qwen",
                "model_path": str(REPO_ROOT / "models" / "checkpoints" / "GeoBox-R1"),
                "adapter_path": None,
            },
        )

    def test_adaptive_tau_uses_the_public_rl_subset_path(self):
        assert_public_rl_path(self, "visualization/analyze_tau_adaptive.py")

    def test_fixed_tau_uses_the_public_rl_subset_path(self):
        assert_public_rl_path(self, "visualization/analyze_tau_fixed.py")

    def test_tau_analyses_match_the_training_adaptive_scale(self):
        adaptive_source = (VISUALIZATION_DIR / "analyze_tau_adaptive.py").read_text(encoding="utf-8")
        fixed_source = (VISUALIZATION_DIR / "analyze_tau_fixed.py").read_text(encoding="utf-8")
        self.assertNotIn("TAU_MIN", adaptive_source)
        self.assertNotIn("TAU_MIN", fixed_source)
        self.assertNotIn(", 1.0)", adaptive_source)
        self.assertNotIn(", 1.0)", fixed_source)
        fixed_ref = assigned_value_node("visualization/analyze_tau_fixed.py", "C_TAU_REF")
        self.assertIsInstance(fixed_ref, ast.Constant)
        self.assertEqual(fixed_ref.value, 8)

    def test_tau_analysis_reward_matches_the_training_reward(self):
        import math
        import numpy as np

        names = {"poly_to_gaussian", "calculate_wd", "calculate_r_wd"}
        training_reward = load_function_subset("training/reward_plugin_qwen3vl.py", names)
        analyses = [
            load_function_subset("visualization/analyze_tau_adaptive.py", names),
            load_function_subset("visualization/analyze_tau_fixed.py", names),
        ]
        target = [[100, 100], [300, 100], [300, 200], [100, 200]]
        prediction = [[125, 110], [325, 110], [325, 210], [125, 210]]
        _, covariance = training_reward.poly_to_gaussian(target)
        c_tau_node = assigned_value_node("training/reward_plugin_qwen3vl.py", "C_TAU")
        self.assertIsInstance(c_tau_node, ast.Constant)
        tau = c_tau_node.value * math.sqrt(float(np.trace(covariance)))
        expected = training_reward.calculate_r_wd(target, prediction, tau=tau)
        for analysis in analyses:
            with self.subTest(analysis=analysis):
                self.assertAlmostEqual(
                    analysis.calculate_r_wd(target, prediction, tau),
                    expected,
                    places=10,
                )

    def test_tau_analyses_report_and_skip_degenerate_boxes(self):
        records = []
        for index, size in enumerate([8, 12, 20, 32, 48, 72, 96, 140, 200, 300] * 3):
            x = 100 + index
            records.append({
                "oriented_bbox": [[x, x], [x + size, x], [x + size, x + size / 2], [x, x + size / 2]],
                "image_width": 1000,
                "image_height": 1000,
                "origin_dataset": f"synthetic-{index % 3}",
            })
        records.append({
            "oriented_bbox": [[0, 0], [10, 0], [10, 0], [0, 0]],
            "image_width": 1000,
            "image_height": 1000,
            "origin_dataset": "degenerate",
        })
        payload = "".join(json.dumps(record) + "\n" for record in records)
        real_open = Path.open

        def fake_open(path, *args, **kwargs):
            if path.name == "rl_obb_20pct.jsonl":
                return io.StringIO(payload)
            return real_open(path, *args, **kwargs)

        for script in ("analyze_tau_adaptive.py", "analyze_tau_fixed.py"):
            with self.subTest(script=script):
                output = io.StringIO()
                with mock.patch.object(Path, "open", fake_open), contextlib.redirect_stdout(output):
                    runpy.run_path(str(VISUALIZATION_DIR / script), run_name="__main__")
                self.assertIn("Skipped 1 degenerate OBB", output.getvalue())

    def test_fixed_tau_reports_insufficient_scale_diversity_clearly(self):
        cases = {
            "one-valid-one-degenerate": [
                {
                    "oriented_bbox": [[0, 0], [20, 0], [20, 10], [0, 10]],
                    "image_width": 1000,
                    "image_height": 1000,
                    "origin_dataset": "valid",
                },
                {
                    "oriented_bbox": [[0, 0], [10, 0], [10, 0], [0, 0]],
                    "image_width": 1000,
                    "image_height": 1000,
                    "origin_dataset": "degenerate",
                },
            ],
            "same-scale": [
                {
                    "oriented_bbox": [[i, i], [i + 20, i], [i + 20, i + 10], [i, i + 10]],
                    "image_width": 1000,
                    "image_height": 1000,
                    "origin_dataset": "same-scale",
                }
                for i in range(6)
            ],
        }
        real_open = Path.open
        for name, records in cases.items():
            payload = "".join(json.dumps(record) + "\n" for record in records)

            def fake_open(path, *args, **kwargs):
                if path.name == "rl_obb_20pct.jsonl":
                    return io.StringIO(payload)
                return real_open(path, *args, **kwargs)

            with self.subTest(case=name), mock.patch.object(Path, "open", fake_open):
                with self.assertRaisesRegex(RuntimeError, "scale groups"):
                    runpy.run_path(
                        str(VISUALIZATION_DIR / "analyze_tau_fixed.py"),
                        run_name="__main__",
                    )

    def test_infer_and_plot_does_not_execute_model_text_with_eval(self):
        tree = ast.parse(
            (VISUALIZATION_DIR / "infer_and_plot.py").read_text(encoding="utf-8")
        )
        eval_calls = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "eval"
        ]
        self.assertEqual(eval_calls, [])

    def test_infer_and_plot_has_no_bare_except_handlers(self):
        tree = ast.parse(
            (VISUALIZATION_DIR / "infer_and_plot.py").read_text(encoding="utf-8")
        )
        bare_handlers = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.ExceptHandler) and node.type is None
        ]
        self.assertEqual(bare_handlers, [])

    def test_infer_and_plot_uses_final_model_and_last_json_candidate(self):
        module = load_module("visualization/infer_and_plot.py")
        self.assertEqual(
            Path(module.DEFAULT_MODEL),
            REPO_ROOT / "models" / "checkpoints" / "GeoBox-R1",
        )
        hbb = (
            'draft {"horizontal_bbox": [0, 0, 10, 10]} '
            'final {"horizontal_bbox": [100, 200, 700, 800]}'
        )
        obb = (
            'draft {"oriented_bbox": [[0, 0], [1, 0], [1, 1], [0, 1]]} '
            'final {"oriented_bbox": [[10, 20], [30, 20], [30, 40], [10, 40]]}'
        )
        self.assertEqual(module.extract_hbb_from_response(hbb), [100, 200, 700, 800])
        self.assertEqual(
            module.extract_obb_from_response(obb),
            [[10, 20], [30, 20], [30, 40], [10, 40]],
        )

    def test_plot_obb_default_output_is_repository_anchored(self):
        module = load_module("visualization/plot_obb_boxes.py")
        self.assertEqual(
            Path(module.DEFAULT_OUTPUT_DIR),
            REPO_ROOT / "output" / "visualizations" / "gt_pred_boxes",
        )

    def test_scheduler_rejects_an_empty_gpu_pool(self):
        module = load_module("visualization/compare_testsets.py")
        args = types.SimpleNamespace(
            geoground_gpu_pool="auto",
            gpu_pool="auto",
            output_root=Path(tempfile.gettempdir()),
            geoground_conda_env="geoground",
        )
        with mock.patch.object(module, "parse_gpu_pool", return_value=[]):
            with self.assertRaisesRegex(RuntimeError, "GPU"):
                module.run_infer_jobs("hbb", [], ["geoground"], args)

    def test_scheduler_rejects_gpu_ids_that_are_not_visible(self):
        module = load_module("visualization/compare_testsets.py")
        args = types.SimpleNamespace(
            geoground_gpu_pool="99",
            gpu_pool="99",
            output_root=Path(tempfile.gettempdir()),
            geoground_conda_env="geoground",
        )
        with mock.patch.object(module, "parse_gpu_pool", return_value=[99]), mock.patch.object(
            module, "query_gpus", return_value=[]
        ):
            with self.assertRaisesRegex(RuntimeError, "visible"):
                module.run_infer_jobs("hbb", [], ["qwen"], args)

    def test_scheduler_rejects_partially_visible_gpu_pools(self):
        module = load_module("visualization/compare_testsets.py")
        state = module.GPUState(0, "GPU", 24000, 0, 24000, 0)
        args = types.SimpleNamespace(
            geoground_gpu_pool="99,0",
            gpu_pool="99,0",
            output_root=Path(tempfile.gettempdir()),
            geoground_conda_env="geoground",
        )
        with mock.patch.object(module, "parse_gpu_pool", return_value=[99, 0]), mock.patch.object(
            module, "query_gpus", return_value=[state]
        ), mock.patch.object(module, "runtime_check_geoground") as runtime_check:
            with self.assertRaisesRegex(RuntimeError, "99"):
                module.run_infer_jobs("hbb", [], ["geoground"], args)
        runtime_check.assert_not_called()

    def test_scheduler_propagates_worker_failures(self):
        module = load_module("visualization/compare_testsets.py")
        spec = module.DatasetSpec("sample", "hbb", Path("sample.jsonl"), "images", "Sample", 1)
        state = module.GPUState(0, "GPU", 24000, 0, 24000, 0)
        process = mock.Mock()
        process.poll.return_value = 3
        log_file = mock.Mock()
        args = types.SimpleNamespace(
            geoground_gpu_pool="0",
            gpu_pool="0",
            max_workers=1,
            batch_policy="fixed",
            fixed_batch_size=1,
            geoground_fixed_batch_size=1,
            min_free_gb=0,
            geoground_min_free_gb=0,
            scheduler_poll_sec=0,
            output_root=Path(tempfile.gettempdir()),
            custom_metainfo_file=None,
        )
        job_info = {
            "proc": process,
            "log_file": log_file,
            "job": module.Job("hbb", "qwen", "sample"),
            "gpu": state,
            "batch_size": 1,
            "log_path": Path("worker.log"),
        }
        with mock.patch.object(module, "parse_gpu_pool", return_value=[0]), mock.patch.object(
            module, "query_gpus", return_value=[state]
        ), mock.patch.object(
            module, "resolve_dataset_spec_for_name", return_value=spec
        ), mock.patch.object(module, "spawn_job", return_value=job_info):
            with self.assertRaisesRegex(RuntimeError, "failed"):
                module.run_infer_jobs("hbb", [spec], ["qwen"], args)

    def test_batch_response_count_must_match_request_count(self):
        module = load_module("visualization/compare_testsets.py")
        with self.assertRaisesRegex(RuntimeError, "Qwen"):
            module.require_matching_batch_size([1, 2], ["one"], "Qwen")

    def test_geoground_compatibility_does_not_disable_argument_validation(self):
        source = (VISUALIZATION_DIR / "compare_testsets.py").read_text(encoding="utf-8")
        self.assertNotIn("_validate_model_kwargs", source)

    def test_infer_and_plot_reports_image_write_failures(self):
        from PIL import Image

        module = load_module("visualization/infer_and_plot.py")
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "source.png"
            output = Path(temp_dir) / "output.png"
            Image.new("RGB", (8, 8)).save(source)
            with mock.patch.object(module.cv2, "imwrite", return_value=False):
                with self.assertRaises(OSError):
                    module.visualize(None, None, str(source), "target", "", str(output))

    def test_visualization_dependencies_are_declared(self):
        requirements = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")
        self.assertRegex(requirements, r"(?m)^opencv-python(?:-headless)?(?:[<>=!~].*)?$")


if __name__ == "__main__":
    unittest.main()
