from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ocr_engine.postal_code import DEFAULT_CACHE_PATH, DEFAULT_DB_DIR, PostalCodeIndex


def main() -> None:
    parser = argparse.ArgumentParser(description="Build compact postal-code TSV cache from DB Kode Wilayah Excel files.")
    parser.add_argument("--db-dir", type=Path, default=DEFAULT_DB_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_CACHE_PATH)
    args = parser.parse_args()

    index = PostalCodeIndex.from_excel_dir(args.db_dir)
    index.write_tsv(args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
