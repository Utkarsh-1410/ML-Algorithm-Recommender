"""Project entrypoint.

This repository's GUI lives in the `ARCSaathi` package (PySide6).

Run:
  - `python main.py`

Optional:
  - `python main.py --diagnose` to quickly check required imports.

Note: This file is intentionally small. The app bootstrap is implemented in
`ARCSaathi/app.py`.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from typing import Iterable


def _module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _run_diagnostics() -> int:
    required = [
        "PySide6",
        "pandas",
        "numpy",
        "sklearn",
    ]

    missing = [name for name in required if not _module_available(name)]
    if missing:
        print("Missing required packages:")
        for name in missing:
            print(f"  - {name}")
        print("\nInstall dependencies with:")
        print("  pip install -r requirements.txt")
        return 2

    print("All required packages look importable.")
    return 0


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Launch the ARCSaathi GUI")
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="Check that core dependencies are installed and importable",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.diagnose:
        return _run_diagnostics()

    try:
        from ARCSaathi.app import run
    except Exception as exc:
        print("Failed to import ARCSaathi.")
        print(f"Error: {exc}")
        print("\nTry:")
        print("  - pip install -r requirements.txt")
        print("  - python main.py --diagnose")
        return 1

    return int(run())


if __name__ == "__main__":
    raise SystemExit(main())
