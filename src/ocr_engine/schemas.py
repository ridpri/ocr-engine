from __future__ import annotations

from dataclasses import asdict, dataclass, field


FieldStatus = str


@dataclass(slots=True)
class FieldResult:
    value: str | None
    confidence: float
    status: FieldStatus
    evidence: list[str] = field(default_factory=list)
    raw: str | None = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class DocumentResult:
    document_type: str
    schema_version: str
    fields: dict[str, FieldResult]
    raw_text: str
    warnings: list[str] = field(default_factory=list)
    engine_version: str = "ocr-engine-poc/0.1.0"

    @property
    def needs_review(self) -> bool:
        return bool(self.warnings) or any(field.status == "invalid" for field in self.fields.values())

    def to_dict(self) -> dict:
        return {
            "document_type": self.document_type,
            "schema_version": self.schema_version,
            "fields": {key: value.to_dict() for key, value in self.fields.items()},
            "needs_review": self.needs_review,
            "warnings": self.warnings,
            "raw_text": self.raw_text,
            "engine_version": self.engine_version,
        }
