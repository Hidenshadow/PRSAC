"""Run the project test suite without external test dependencies."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


def main() -> int:
    project_root = Path(__file__).resolve().parent
    suite = unittest.defaultTestLoader.discover(str(project_root / "tests"))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
