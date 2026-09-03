"""Tests for training/: the launchers are exercised with a fake `swift` on PATH."""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from _support import REPO_ROOT, TRAINING_DIR


class TrainingScriptTests(unittest.TestCase):
    def _run_with_fake_swift(self, script: str, *arguments: str, env_overrides=None):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            capture_path = temp_path / "swift-call.txt"
            swift_path = temp_path / "swift"
            swift_path.write_text(
                '#!/bin/sh\n{ pwd; printf "%s\\n" "$@"; } > "$CAPTURE_FILE"\n',
                encoding="utf-8",
            )
            swift_path.chmod(0o755)
            env = os.environ.copy()
            env["PATH"] = f"{temp_path}{os.pathsep}{env['PATH']}"
            env["CAPTURE_FILE"] = str(capture_path)
            env.update(env_overrides or {})
            process = subprocess.run(
                ["bash", f"training/{script}", *arguments],
                cwd=REPO_ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            captured = capture_path.read_text(encoding="utf-8").splitlines() if capture_path.exists() else []
            return process, captured

    @staticmethod
    def _keyed_arguments(captured):
        arguments = captured[1:]
        return {
            argument: arguments[index + 1]
            for index, argument in enumerate(arguments[:-1])
            if argument.startswith("--")
        }

    def test_training_launchers_run_from_the_repository_root(self):
        """Image paths inside the training JSONL are repository-root relative and ms-swift
        opens them relative to the working directory, so every launcher must run from the
        repository root and every path argument must resolve inside the repository."""
        path_flags = {"--model", "--dataset", "--external_plugins", "--output_dir"}
        expected_datasets = {
            "sft.sh": REPO_ROOT / "data" / "GeoBox-R1-Data" / "sft" / "sft_curriculum_cot.jsonl",
            "gdpo.sh": REPO_ROOT / "data" / "GeoBox-R1-Data" / "rl" / "rl_obb_20pct.jsonl",
            "gdpo_fixed_tau.sh": REPO_ROOT / "data" / "GeoBox-R1-Data" / "rl" / "rl_obb_20pct.jsonl",
            "gdpo_qwen2_5vl.sh": REPO_ROOT / "data" / "GeoBox-R1-Data" / "rl" / "rl_obb_20pct.jsonl",
        }
        for script in (
            "sft.sh",
            "gdpo.sh",
            "gdpo_fixed_tau.sh",
            "gdpo_qwen2_5vl.sh",
            "rollout_vllm.sh",
        ):
            with self.subTest(script=script):
                process, captured = self._run_with_fake_swift(script)
                self.assertEqual(process.returncode, 0, process.stderr)
                self.assertTrue(captured, f"{script} did not invoke swift")
                invocation_cwd = Path(captured[0]).resolve()
                self.assertEqual(invocation_cwd, REPO_ROOT, f"{script} must run from the repository root")
                arguments = self._keyed_arguments(captured)
                for argument, value in arguments.items():
                    if argument not in path_flags:
                        continue
                    # Lexical check: a symlinked models/ or data/ directory still counts as inside.
                    resolved = Path(os.path.normpath(invocation_cwd / value))
                    self.assertTrue(
                        resolved.is_relative_to(REPO_ROOT),
                        f"{script}: {argument} resolves outside the repository: {resolved}",
                    )
                if script in expected_datasets:
                    self.assertEqual(Path(os.path.normpath(invocation_cwd / arguments["--dataset"])), expected_datasets[script])
                if "--external_plugins" in arguments:
                    plugin = (invocation_cwd / arguments["--external_plugins"]).resolve()
                    self.assertTrue(plugin.is_file(), f"{script}: reward plugin not found: {plugin}")
                elif script.startswith("gdpo"):
                    self.fail(f"{script} must load a reward plugin via --external_plugins")

    def test_sft_disables_both_dataset_and_dataloader_shuffle(self):
        source = (TRAINING_DIR / "sft.sh").read_text(encoding="utf-8")
        self.assertRegex(source, r"--dataset_shuffle\s+false(?:\s|\\)")
        self.assertRegex(source, r"--train_dataloader_shuffle\s+false(?:\s|\\)")

    def test_qwen2_5_rollout_matches_the_gdpo_launcher(self):
        rollout_process, rollout_call = self._run_with_fake_swift("rollout_vllm.sh", "qwen2_5")
        gdpo_process, gdpo_call = self._run_with_fake_swift("gdpo_qwen2_5vl.sh")
        self.assertEqual(rollout_process.returncode, 0, rollout_process.stderr)
        self.assertEqual(gdpo_process.returncode, 0, gdpo_process.stderr)
        rollout_args = self._keyed_arguments(rollout_call)
        gdpo_args = self._keyed_arguments(gdpo_call)
        self.assertEqual(rollout_args["--model"], gdpo_args["--model"])
        self.assertEqual(rollout_args["--model_type"], gdpo_args["--model_type"])
        self.assertEqual(rollout_args["--norm_bbox"], gdpo_args["--norm_bbox"])
        self.assertEqual(rollout_args["--port"], gdpo_args["--vllm_server_port"])

    def test_fixed_tau_matches_main_gdpo_training_hyperparameters(self):
        main_process, main_call = self._run_with_fake_swift("gdpo.sh")
        fixed_process, fixed_call = self._run_with_fake_swift("gdpo_fixed_tau.sh")
        self.assertEqual(main_process.returncode, 0, main_process.stderr)
        self.assertEqual(fixed_process.returncode, 0, fixed_process.stderr)
        main_args = self._keyed_arguments(main_call)
        fixed_args = self._keyed_arguments(fixed_call)
        shared_hyperparameters = (
            "--model",
            "--model_type",
            "--template",
            "--dataset",
            "--tuner_type",
            "--lora_rank",
            "--lora_alpha",
            "--attn_impl",
            "--torch_dtype",
            "--max_length",
            "--max_completion_length",
            "--num_train_epochs",
            "--per_device_train_batch_size",
            "--learning_rate",
            "--gradient_accumulation_steps",
            "--warmup_ratio",
            "--dataloader_num_workers",
            "--num_generations",
            "--temperature",
            "--deepspeed",
            "--beta",
            "--num_iterations",
            "--max_grad_norm",
            "--scale_rewards",
        )
        for argument in shared_hyperparameters:
            with self.subTest(argument=argument):
                self.assertEqual(main_args.get(argument), fixed_args.get(argument))

    def test_merge_lora_supports_qwen2_5_gdpo(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_root = Path(temp_dir)
            older = run_root / "v0-20260101-000000" / "checkpoint-100"
            latest = run_root / "v1-20260102-000000" / "checkpoint-20"
            for checkpoint in (older, latest):
                checkpoint.mkdir(parents=True)
                (checkpoint / "adapter_config.json").write_text("{}", encoding="utf-8")
            os.utime(older, (100, 100))
            os.utime(latest, (200, 200))
            process, captured = self._run_with_fake_swift(
                "merge_lora.sh",
                "gdpo-qwen2_5",
                env_overrides={"ADAPTER_ROOT": str(run_root)},
            )
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertTrue(captured, "merge_lora.sh did not invoke swift")
        self.assertEqual(Path(captured[0]).resolve(), REPO_ROOT)
        keyed_arguments = self._keyed_arguments(captured)
        self.assertEqual(keyed_arguments["--model_type"], "qwen2_5_vl")
        self.assertEqual(keyed_arguments["--template"], "qwen2_5_vl")
        self.assertEqual(
            (Path(captured[0]) / keyed_arguments["--model"]).resolve(),
            REPO_ROOT / "models" / "checkpoints" / "GeoBox-R1-SFT-Qwen2_5VL",
        )
        self.assertEqual(
            Path(keyed_arguments["--adapters"]).resolve(),
            latest,
        )

    def test_merge_lora_fails_when_no_adapter_checkpoint_exists(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            missing_root = Path(temp_dir) / "missing"
            process, captured = self._run_with_fake_swift(
                "merge_lora.sh",
                "gdpo",
                env_overrides={"ADAPTER_ROOT": str(missing_root)},
            )
        self.assertNotEqual(process.returncode, 0)
        self.assertEqual(captured, [])
        self.assertIn("no adapter checkpoint found", process.stderr)

    def test_merge_lora_explicit_adapter_bypasses_run_discovery(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            adapter = temp_path / "explicit-adapter"
            adapter.mkdir()
            (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
            process, captured = self._run_with_fake_swift(
                "merge_lora.sh",
                "gdpo",
                env_overrides={
                    "ADAPTER": str(adapter),
                    "ADAPTER_ROOT": str(temp_path / "missing"),
                },
            )
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(
            Path(self._keyed_arguments(captured)["--adapters"]).resolve(),
            adapter,
        )


if __name__ == "__main__":
    unittest.main()
