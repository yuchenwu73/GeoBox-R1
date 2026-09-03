"""Tests for evaluation/: prompts, output parsers, norm1000 de-normalisation and IoU, checked
against the demo/ mirrors of the same functions so the two cannot drift apart."""

import sys
import types
import unittest
from unittest import mock

from _support import RectPolygon, load_module


class PromptAndGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.demo = load_module("demo/inference.py")
        cls.hbb = load_module("evaluation/evaluate_hbb.py")
        cls.obb = load_module("evaluation/evaluate_obb.py")

    def test_demo_hbb_prompt_matches_evaluation_prompt(self):
        question = "the aircraft beside the hangar"
        self.assertEqual(self.demo.hbb_prompt(question), self.hbb.get_prompt(question))

    def test_demo_obb_prompt_matches_evaluation_prompt(self):
        question = "the ship pointing northeast"
        self.assertEqual(self.demo.obb_prompt(question), self.obb.get_obb_prompt(question))

    def test_hbb_parsers_accept_fenced_model_output(self):
        output = '```json\n[{"horizontal_bbox": [100, 200, 700, 800]}]\n```'
        expected = [100.0, 200.0, 700.0, 800.0]
        self.assertEqual(self.demo.parse_hbb(output), expected)
        self.assertEqual(self.hbb.parse_bbox_from_output(output), expected)

    def test_obb_parsers_use_the_last_valid_json_candidate(self):
        output = (
            'draft [{"oriented_bbox": [[0, 0], [1, 0], [1, 1], [0, 1]]}] '
            'final [{"oriented_bbox": [[10, 20], [30, 20], [30, 40], [10, 40]]}]'
        )
        expected = [[10.0, 20.0], [30.0, 20.0], [30.0, 40.0], [10.0, 40.0]]
        self.assertEqual(self.demo.parse_obb(output), expected)
        self.assertEqual(self.obb.parse_obb_from_output(output), expected)

    def test_hbb_norm1000_coordinates_denormalize_to_original_pixels(self):
        bbox = [100, 200, 700, 800]
        expected = [200.0, 200.0, 1400.0, 800.0]
        self.assertEqual(self.demo.denorm_hbb(bbox, (2000, 1000)), expected)
        self.assertEqual(
            self.hbb._maybe_denormalize_bbox(bbox, (2000, 1000), "norm1000"),
            expected,
        )

    def test_obb_norm1000_coordinates_denormalize_to_original_pixels(self):
        obb = [[100, 200], [700, 200], [700, 800], [100, 800]]
        expected = [[200.0, 200.0], [1400.0, 200.0], [1400.0, 800.0], [200.0, 800.0]]
        self.assertEqual(self.demo.denorm_obb(obb, (2000, 1000)), expected)
        self.assertEqual(
            self.obb._maybe_denormalize_obb(obb, (2000, 1000), "norm1000"),
            expected,
        )

    def test_hbb_iou_is_one_third_for_half_overlapping_equal_boxes(self):
        first = [0, 0, 10, 10]
        second = [5, 0, 15, 10]
        self.assertAlmostEqual(self.demo.iou_hbb(first, second), 1 / 3)
        self.assertAlmostEqual(self.hbb.calculate_iou(first, second), 1 / 3)

    def test_obb_iou_is_one_third_for_half_overlapping_equal_rectangles(self):
        first = [[0, 0], [10, 0], [10, 10], [0, 10]]
        second = [[5, 0], [15, 0], [15, 10], [5, 10]]
        self.obb.Polygon = RectPolygon
        geometry_module = types.ModuleType("shapely.geometry")
        geometry_module.Polygon = RectPolygon
        shapely_module = types.ModuleType("shapely")
        shapely_module.geometry = geometry_module
        with mock.patch.dict(
            sys.modules,
            {"shapely": shapely_module, "shapely.geometry": geometry_module},
        ):
            self.assertAlmostEqual(self.demo.iou_obb(first, second), 1 / 3)
        self.assertAlmostEqual(self.obb.calculate_rotated_iou(first, second), 1 / 3)


if __name__ == "__main__":
    unittest.main()
