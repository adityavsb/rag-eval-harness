#!/usr/bin/env python3
"""Print SHA-256 for local report JSON (audit only; never commit full traces)."""

import argparse
import hashlib
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="SHA-256 checksum for a local eval report JSON file.")
    parser.add_argument("paths", nargs="+", help="Path(s) to local report JSON under reports/")
    args = parser.parse_args()

    for raw in args.paths:
        path = Path(raw)
        if not path.is_file():
            print(f"{path}: not found", file=sys.stderr)
            sys.exit(1)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        print(f"{digest}  {path}")


if __name__ == "__main__":
    main()
