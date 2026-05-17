from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Protocol


class OcrDependencyError(RuntimeError):
    """Raised when an optional OCR backend is not installed or cannot start."""


@dataclass(slots=True)
class OcrToken:
    text: str
    confidence: float
    bbox: list | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class OcrResult:
    raw_text: str
    tokens: list[OcrToken] = field(default_factory=list)
    provider: str = "unknown"

    def to_dict(self) -> dict:
        return {
            "raw_text": self.raw_text,
            "provider": self.provider,
            "tokens": [token.to_dict() for token in self.tokens],
        }


class OcrProvider(Protocol):
    def extract_text(self, image_path: str) -> OcrResult:
        """Extract OCR text and tokens from an image path."""
