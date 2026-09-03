# Lets `python -m unittest tests.test_x` work as well as `discover -s tests`.
import sys
from pathlib import Path

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
