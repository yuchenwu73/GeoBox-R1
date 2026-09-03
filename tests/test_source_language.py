"""Code, comments and runtime messages are English; only the demo UI strings are bilingual."""

import ast
import io
import tokenize
import unittest

from _support import (
    BASELINES_DIR,
    CJK_RE,
    DATA_PIPELINE_DIR,
    DEMO_DIR,
    EVALUATION_DIR,
    REPO_ROOT,
    TRAINING_DIR,
    VISUALIZATION_DIR,
)


class SourceLanguageTests(unittest.TestCase):
    def test_code_directories_have_no_cjk_text(self):
        offenders = []
        code_dirs = (EVALUATION_DIR, TRAINING_DIR, VISUALIZATION_DIR, BASELINES_DIR, DATA_PIPELINE_DIR)
        for directory in code_dirs:
            for path in sorted(directory.rglob("*")):
                if path.suffix not in {".py", ".sh"}:
                    continue
                for line_number, line in enumerate(
                    path.read_text(encoding="utf-8").splitlines(), start=1
                ):
                    if CJK_RE.search(line):
                        offenders.append(f"{path.relative_to(REPO_ROOT)}:{line_number}")
        self.assertEqual(offenders, [])

    def test_demo_python_comments_and_docstrings_have_no_cjk_text(self):
        offenders = []
        for path in sorted(DEMO_DIR.glob("*.py")):
            source = path.read_text(encoding="utf-8")
            tokens = tokenize.generate_tokens(io.StringIO(source).readline)
            offenders.extend(
                f"{path.relative_to(REPO_ROOT)}:{token.start[0]}"
                for token in tokens
                if token.type == tokenize.COMMENT and CJK_RE.search(token.string)
            )
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                docstring = ast.get_docstring(node, clean=False)
                if docstring and CJK_RE.search(docstring):
                    line_number = node.body[0].lineno if node.body else node.lineno
                    offenders.append(f"{path.relative_to(REPO_ROOT)}:{line_number}")
        self.assertEqual(offenders, [])

    def test_demo_shell_comments_have_no_cjk_text(self):
        offenders = []
        for path in sorted(DEMO_DIR.glob("*.sh")):
            for line_number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                if line.lstrip().startswith("#") and CJK_RE.search(line):
                    offenders.append(f"{path.relative_to(REPO_ROOT)}:{line_number}")
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
