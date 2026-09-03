"""End-to-end test of data_pipeline/ on a tiny synthetic refGeo tree."""
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = REPO_ROOT / "data_pipeline"


def load_script(name):
    """Import a pipeline script from its file path (they are scripts, not a package)."""
    if str(PIPELINE_DIR) not in sys.path:
        sys.path.insert(0, str(PIPELINE_DIR))  # build_sft imports build_obb_cot
    spec = importlib.util.spec_from_file_location(f"pipeline_{name}", PIPELINE_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_main(module, argv):
    old = sys.argv
    sys.argv = [module.__name__, *argv]
    try:
        module.main()
    finally:
        sys.argv = old


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


HBB_SUBSETS = {"RSVG": "rsvg_train.jsonl", "DIOR-RSVG": "dior_rsvg_train.jsonl"}
OBB_SUBSETS = {"AVVG": "avvg_train.jsonl", "VRSBench": "vrsbench_train.jsonl", "GeoChat": "geochat_train.jsonl"}
N_HBB = 4   # records per HBB subset
N_OBB = 8   # records per OBB subset -> 6 OBB + 2 CoT after the 75/25 split


def make_refgeo(root: Path):
    """Write a miniature refGeo tree: metainfo JSONL plus 1x1 images per subset."""
    (root / "metainfo").mkdir(parents=True)
    for subset, meta in HBB_SUBSETS.items():
        (root / "images" / subset).mkdir(parents=True)
        with open(root / "metainfo" / meta, "w", encoding="utf-8") as f:
            for i in range(N_HBB):
                Image.new("RGB", (64, 32)).save(root / "images" / subset / f"{subset}_{i}.jpg")
                f.write(json.dumps({"image_id": f"{subset}_{i}.jpg", "question": f"{subset} target {i}",
                                    "bbox": [1.5, 2.5, 10.4, 20.6]}) + "\n")
    for subset, meta in OBB_SUBSETS.items():
        (root / "images" / subset).mkdir(parents=True)
        with open(root / "metainfo" / meta, "w", encoding="utf-8") as f:
            for i in range(N_OBB):
                # GeoChat metainfo says .jpg while the image on disk is .png (as in refGeo).
                ext = ".png" if subset == "GeoChat" else ".jpg"
                Image.new("RGB", (100, 50)).save(root / "images" / subset / f"{subset}_{i}{ext}")
                f.write(json.dumps({"image_id": f"{subset}_{i}.jpg", "question": f"{subset} target {i}",
                                    "bbox": [10, 10, 30, 20],
                                    "poly": [[10.4, 10.5], [30, 10], [30, 20], [10, 20]]}) + "\n")


class DataPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name) / "refGeo"
        cls.out = Path(cls.tmp.name) / "GeoBox-R1-Data"
        make_refgeo(cls.root)
        cls.build_hbb = load_script("build_hbb")
        cls.build_obb_cot = load_script("build_obb_cot")
        cls.build_sft = load_script("build_sft")
        cls.build_rl = load_script("build_rl")
        cwd = os.getcwd()
        os.chdir(cls.tmp.name)  # image paths are resolved relative to the working directory
        try:
            run_main(cls.build_hbb, ["--refgeo_root", "refGeo"])
            run_main(cls.build_obb_cot, ["--refgeo_root", "refGeo"])
            run_main(cls.build_sft, ["--refgeo_root", "refGeo", "--output_dir", "GeoBox-R1-Data", "--config", "all"])
            run_main(cls.build_rl, ["--refgeo_root", "refGeo", "--output_dir", "GeoBox-R1-Data", "--percent", "50"])
        finally:
            os.chdir(cwd)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_hbb_records(self):
        for subset in HBB_SUBSETS:
            records = read_jsonl(self.root / "SFT" / f"{subset}_HBB_train.jsonl")
            self.assertEqual(len(records), N_HBB)
            for record in records:
                self.assertEqual(record["origin_dataset"], subset)
                self.assertEqual(record["images"], [f"refGeo/images/{subset}/{subset}_{records.index(record)}.jpg"])
                self.assertEqual(record["objects"]["bbox"], [[2, 3, 10, 21]])  # half-up rounding
                self.assertIn("horizontal_bbox", record["messages"][1]["content"])
                self.assertIn("<ref-object>", record["messages"][0]["content"])

    def test_obb_and_cot_records_and_split(self):
        for subset in OBB_SUBSETS:
            obb = read_jsonl(self.root / "SFT" / f"{subset}_OBB_train.jsonl")
            cot = read_jsonl(self.root / "SFT" / f"{subset}_CoT_train.jsonl")
            self.assertEqual((len(obb), len(cot)), (6, 2))
            for record in obb:
                self.assertEqual(record["objects"]["bbox"], [[10, 11], [30, 10], [30, 20], [10, 20]])
                self.assertIn("oriented_bbox", record["messages"][1]["content"])
            for record in cot:
                self.assertEqual(record["objects"]["bbox"][0], [10, 10, 30, 20])
                self.assertEqual(len(record["objects"]["bbox"]), 5)
                self.assertIn("horizontal_bbox", record["messages"][1]["content"])
                self.assertIn("oriented_bbox", record["messages"][1]["content"])
                self.assertIn("stepwise", record["messages"][0]["content"])
            ext = ".png" if subset == "GeoChat" else ".jpg"
            self.assertTrue(all(r["images"][0].endswith(ext) for r in obb + cot))

    def test_obb_split_is_deterministic(self):
        first = read_jsonl(self.root / "SFT" / "AVVG_OBB_train.jsonl")
        with tempfile.TemporaryDirectory() as other:
            make_refgeo(Path(other) / "refGeo")
            cwd = os.getcwd()
            os.chdir(other)
            try:
                run_main(self.build_obb_cot, ["--refgeo_root", "refGeo"])
            finally:
                os.chdir(cwd)
            again = read_jsonl(Path(other) / "refGeo" / "SFT" / "AVVG_OBB_train.jsonl")
        self.assertEqual([r["objects"]["ref"] for r in first], [r["objects"]["ref"] for r in again])

    def test_sft_sets(self):
        sets = {name: read_jsonl(self.out / "sft" / f"sft_{name}.jsonl") for name in self.build_sft.CONFIGS}
        total = 2 * N_HBB + 3 * N_OBB
        for records in sets.values():
            self.assertEqual(len(records), total)

        def kind(record):
            answer = record["messages"][1]["content"]
            if "horizontal_bbox" in answer and "oriented_bbox" in answer:
                return "cot"
            return "obb" if "oriented_bbox" in answer else "hbb"

        curriculum = [kind(r) for r in sets["curriculum_cot"]]
        self.assertEqual(curriculum, ["hbb"] * 8 + ["obb"] * 18 + ["cot"] * 6)
        self.assertEqual([kind(r) for r in sets["curriculum_no_cot"]], ["hbb"] * 8 + ["obb"] * 24)
        for record in sets["curriculum_no_cot"][8:]:
            self.assertEqual(len(record["objects"]["bbox"]), 4)  # CoT converted to a plain OBB record
        for record in sets["curriculum_no_cot"][-6:]:  # converted records use the released prompt verbatim
            self.assertEqual(record["messages"][0]["content"], self.build_sft.CONVERTED_OBB_PROMPT)

        def key(record):
            return json.dumps(record, sort_keys=True)

        self.assertEqual(sorted(map(key, sets["mixed_cot"])), sorted(map(key, sets["curriculum_cot"])))
        self.assertNotEqual([kind(r) for r in sets["mixed_cot"]], curriculum)
        self.assertEqual(sorted(map(key, sets["mixed_no_cot"])), sorted(map(key, sets["curriculum_no_cot"])))

    def test_rl_records_and_subset(self):
        for subset in OBB_SUBSETS:
            records = read_jsonl(self.root / "RL" / f"{subset}_OBB_train.jsonl")
            self.assertEqual(len(records), 6)
            for record in records:
                self.assertEqual(record["oriented_bbox"], record["objects"]["bbox"])
                self.assertEqual((record["image_width"], record["image_height"]), (100, 50))
        subset = read_jsonl(self.out / "rl" / "rl_obb_50pct.jsonl")
        self.assertEqual(len(subset), 9)  # 3 per dataset
        self.assertEqual([r["origin_dataset"] for r in subset], ["AVVG"] * 3 + ["GeoChat"] * 3 + ["VRSBench"] * 3)

    def test_rl_subset_is_deterministic(self):
        first = read_jsonl(self.out / "rl" / "rl_obb_50pct.jsonl")
        cwd = os.getcwd()
        os.chdir(self.tmp.name)
        try:
            run_main(self.build_rl, ["--refgeo_root", "refGeo", "--output_dir", "again",
                                     "--percent", "50", "--skip_convert"])
        finally:
            os.chdir(cwd)
        again = read_jsonl(Path(self.tmp.name) / "again" / "rl" / "rl_obb_50pct.jsonl")
        self.assertEqual(first, again)


if __name__ == "__main__":
    unittest.main()
