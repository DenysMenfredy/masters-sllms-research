"""Version-one benchmark semantics: verbatim values and duplicate-preserving slots."""

import json
from collections import Counter

from pydantic import BaseModel, ConfigDict, Field


class Slot(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    type: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    value: str = Field(min_length=1)


class SemanticAction(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    intent: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    slots: list[Slot]

    def slot_multiset(self) -> Counter[tuple[str, str]]:
        return Counter((slot.type, slot.value) for slot in self.slots)


def _unique_keys(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_prediction(raw: str) -> SemanticAction:
    """Reject incomplete JSON, duplicate keys, extra fields, and non-string slots."""
    return SemanticAction.model_validate(json.loads(raw, object_pairs_hook=_unique_keys))
