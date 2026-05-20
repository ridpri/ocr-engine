from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ocr_engine import cli_eval  # noqa: E402


RECORDS_BASENAME = "records.jsonl"
SUMMARY_BASENAME = "summary.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an STNK benchmark over a local sample folder.")
    parser.add_argument("--input", required=True, help="Path to STNK image/PDF file or directory.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where benchmark JSONL and summary files are written.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum files to process from a folder. Defaults to the full corpus if omitted.",
    )
    parser.add_argument(
        "--mode",
        default="accurate",
        choices=["fast", "accurate"],
        help="OCR processing mode.",
    )
    return parser.parse_args(argv)


@contextlib.contextmanager
def _patched_argv(argv: list[str]):
    old_argv = list(sys.argv)
    sys.argv = argv
    try:
        yield
    finally:
        sys.argv = old_argv


def run_stnk_benchmark(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    limit: int | None = None,
    mode: str = "accurate",
) -> tuple[Path, Path]:
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = output_dir / RECORDS_BASENAME
    summary_path = output_dir / SUMMARY_BASENAME

    args = [
        "benchmark_stnk.py",
        "--input",
        str(input_path),
        "--document-type",
        "STNK",
        "--mode",
        mode,
        "--jsonl",
        str(jsonl_path),
        "--summary-json",
        str(summary_path),
    ]

    if limit is not None:
        args += ["--limit", str(limit)]
    else:
        args += ["--limit", "0"]

    with _patched_argv(args):
        result = cli_eval.main()

    if result != 0:
        raise RuntimeError(f"cli_eval failed with exit code {result}")

    return jsonl_path, summary_path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_stnk_benchmark(args.input, args.output_dir, limit=args.limit, mode=args.mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
