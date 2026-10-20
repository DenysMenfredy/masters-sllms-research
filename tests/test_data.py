import json
from pathlib import Path

import pytest

from sllm_research.data import audit_dataset, convert_record

ROOT = Path(__file__).resolve().parents[1]


def record(identifier="1", partition="train"):
    return {
        "id": identifier,
        "locale": "en-US",
        "partition": partition,
        "intent": "alarm_set",
        "utt": "wake me at five am and five am",
        "annot_utt": "wake me at [time : five am] and [time : five am]",
    }


def test_converter_preserves_slots_and_official_partition():
    converted = convert_record(record(partition="dev"))
    assert converted["split"] == "dev"
    assert converted["source_id"] == "1"
    assert converted["target"]["slots"] == [
        {"type": "time", "value": "five am"},
        {"type": "time", "value": "five am"},
    ]


def test_converter_keeps_no_slot_requests():
    source = {**record(), "utt": "wake me", "annot_utt": "wake me"}
    assert convert_record(source)["target"]["slots"] == []


def test_annotation_mismatch_fails_instead_of_fabricating_gold():
    with pytest.raises(ValueError, match="round trip"):
        convert_record({**record(), "utt": "a different utterance"})


def test_audit_keeps_splits_and_withholds_test_targets(tmp_path):
    source = tmp_path / "en-US.jsonl"
    source.write_text(
        "\n".join(
            json.dumps(record(str(i), split)) for i, split in enumerate(["train", "dev", "test"])
        )
    )
    output = tmp_path / "audit"
    report = audit_dataset(source, ROOT / "configs/data/massive.yaml", output)
    assert report["counts"]["test"]["examples"] == 1
    assert report["cross_split_exact_duplicate_groups"]
    assert not (output / "test.jsonl").exists()
    assert json.loads((output / "test.manifest.json").read_text())["example_ids"] == ["en-US:2"]
    assert all(
        json.loads(line)["split"] != "test"
        for line in (output / "manual-review.train-dev.jsonl").read_text().splitlines()
    )
    assert len(report["source_sha256"]) == 64
    with pytest.raises(FileExistsError):
        audit_dataset(source, ROOT / "configs/data/massive.yaml", output)


def test_split_source_id_leakage_is_rejected(tmp_path):
    source = tmp_path / "en-US.jsonl"
    source.write_text(json.dumps(record()) + "\n" + json.dumps(record(partition="test")))
    with pytest.raises(ValueError, match="Repeated source ID"):
        audit_dataset(source, ROOT / "configs/data/massive.yaml", tmp_path / "audit")
