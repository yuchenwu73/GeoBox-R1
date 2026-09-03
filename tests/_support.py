"""Shared paths and loaders for the test suite.

Scripts are imported by file path so that the test suite never depends on the
model runtimes (ms-swift, torch, shapely) being installed.
"""

import ast
import importlib.util
import re
import sys
import types
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_DIR = REPO_ROOT / "evaluation"
TRAINING_DIR = REPO_ROOT / "training"
VISUALIZATION_DIR = REPO_ROOT / "visualization"
DEMO_DIR = REPO_ROOT / "demo"
BASELINES_DIR = REPO_ROOT / "baselines"
DATA_PIPELINE_DIR = REPO_ROOT / "data_pipeline"
CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")


def load_module(relative_path: str):
    path = REPO_ROOT / relative_path
    module_name = "repo_test_" + relative_path.replace("/", "_").replace(".", "_")
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def load_function_subset(relative_path: str, names):
    """Load pure functions without executing a script's module-level analysis."""
    path = REPO_ROOT / relative_path
    tree = ast.parse(path.read_text(encoding="utf-8"))
    body = [
        node
        for node in tree.body
        if isinstance(node, ast.Import)
        or isinstance(node, ast.ImportFrom)
        and (node.module or "").split(".")[0] not in {"swift", "shapely"}
        or isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Constant)
        or isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names
    ]
    namespace = {"__file__": str(path)}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)
    return types.SimpleNamespace(**{name: namespace[name] for name in names})


def assigned_value_node(relative_path: str, variable: str) -> ast.expr:
    tree = ast.parse((REPO_ROOT / relative_path).read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id == variable for target in targets):
                return node.value
    raise AssertionError(f"{variable} is not assigned in {relative_path}")


def assert_public_rl_path(test_case, relative_path: str):
    expected = "data/GeoBox-R1-Data/rl/rl_obb_20pct.jsonl"
    value = assigned_value_node(relative_path, "DATA_PATH")
    if isinstance(value, ast.Constant) and isinstance(value.value, str):
        test_case.assertEqual(value.value, expected)
        return
    expression = ast.unparse(value)
    normalized = expression.replace("' / '", "/").replace('" / "', "/")
    for component in expected.split("/"):
        test_case.assertIn(component, normalized)
    test_case.assertTrue(
        any(anchor in expression for anchor in ("REPO_ROOT", "ROOT", "REPO_DIR", "__file__")),
        f"DATA_PATH must be anchored to the repository: {expression}",
    )


class Area:
    def __init__(self, area):
        self.area = area


class RectPolygon:
    """Minimal rectangle-only Polygon used to keep the tests dependency-free."""

    def __init__(self, points):
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        self.bounds = min(xs), min(ys), max(xs), max(ys)
        self.area = max(0.0, self.bounds[2] - self.bounds[0]) * max(
            0.0, self.bounds[3] - self.bounds[1]
        )
        self.is_valid = len(points) >= 3 and self.area > 0

    def intersection(self, other):
        left = max(self.bounds[0], other.bounds[0])
        top = max(self.bounds[1], other.bounds[1])
        right = min(self.bounds[2], other.bounds[2])
        bottom = min(self.bounds[3], other.bounds[3])
        return Area(max(0.0, right - left) * max(0.0, bottom - top))

    def union(self, other):
        intersection = self.intersection(other).area
        return Area(self.area + other.area - intersection)
