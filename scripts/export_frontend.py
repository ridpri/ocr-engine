from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ocr_engine.frontend import frontend_html


def main() -> None:
    public_dir = ROOT / "public"
    public_dir.mkdir(exist_ok=True)
    (public_dir / "index.html").write_text(frontend_html(), encoding="utf-8")


if __name__ == "__main__":
    main()
