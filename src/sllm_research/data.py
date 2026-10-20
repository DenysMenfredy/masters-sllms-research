"""Convert official MASSIVE annotations without inventing missing arguments."""

import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from sllm_research.io import read_jsonl, sha256_file, write_json, write_jsonl
from sllm_research.semantics import SemanticAction, Slot

CONVERTER_VERSION = "massive-annot-v1"
ANNOTATION = re.compile(r"\[([a-z][a-z0-9_]*) : ([^\[\]]+)\]")
SPLITS = ("train", "dev", "test")


def convert_record(record: dict) -> dict:
    """Retain official IDs and splits; verify annotation removal recreates the utterance."""
    for key in ("id", "locale", "partition", "intent", "utt", "annot_utt"):
        if key not in record or not isinstance(record[key], str):
            raise ValueError(f"MASSIVE field {key!r} must be present and a string")
    if record["partition"] not in SPLITS:
        raise ValueError(f"Unknown official partition: {record['partition']}")
    annotated = record["annot_utt"]
    reconstructed = ANNOTATION.sub(lambda match: match[2], annotated)
    # Fail for manual review instead of silently discarding unfamiliar annotation syntax.
    if "[" in reconstructed or "]" in reconstructed or reconstructed != record["utt"]:
        raise ValueError(f"Annotation round trip failed for source ID {record['id']}")
    target = SemanticAction(
        intent=record["intent"],
        slots=[Slot(type=match[1], value=match[2]) for match in ANNOTATION.finditer(annotated)],
    )
    return {
        "id": f"{record['locale']}:{record['id']}",
        "source_id": record["id"],
        "locale": record["locale"],
        "split": record["partition"],
        "utterance": record["utt"],
        "target": target.model_dump(),
    }


def audit_dataset(source: Path, config_path: Path, output: Path) -> dict:
    """Write draft manifests and train/dev targets; test targets are not materialized."""
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["converter_version"] != CONVERTER_VERSION:
        raise ValueError("Configured converter version does not match the implementation")
    intents = config["intents"]
    if not intents or len(intents) != len(set(intents)):
        raise ValueError("The intent allowlist must be nonempty and unique")
    converted = []
    seen_ids = set()
    for record in read_jsonl(source):
        if record.get("locale") != config["locale"] or record.get("intent") not in intents:
            continue
        example = convert_record(record)
        if example["id"] in seen_ids:
            raise ValueError(f"Repeated source ID (including across splits): {example['id']}")
        seen_ids.add(example["id"])
        converted.append(example)
    if not converted:
        raise ValueError("No examples match the configured locale and intent allowlist")
    converted.sort(key=lambda example: example["id"])
    counts = {}
    for split in SPLITS:
        examples = [example for example in converted if example["split"] == split]
        counts[split] = {
            "examples": len(examples),
            "intents": dict(sorted(Counter(e["target"]["intent"] for e in examples).items())),
            "slot_types": dict(
                sorted(
                    Counter(slot["type"] for e in examples for slot in e["target"]["slots"]).items()
                )
            ),
            "slot_combinations": dict(
                sorted(
                    Counter(
                        "+".join(sorted(slot["type"] for slot in e["target"]["slots"])) or "<empty>"
                        for e in examples
                    ).items()
                )
            ),
        }
    by_text = defaultdict(list)
    for example in converted:
        by_text[example["utterance"]].append({"id": example["id"], "split": example["split"]})
    exact_duplicates = [group for group in by_text.values() if len(group) > 1]
    provenance = {
        "status": "draft_not_frozen",
        "dataset": config["dataset"],
        "release": config["release"],
        "locale": config["locale"],
        "source_sha256": sha256_file(source),
        "config_sha256": sha256_file(config_path),
        "allowlist_version": config["allowlist_version"],
        "converter_version": CONVERTER_VERSION,
    }
    report = {
        **provenance,
        "counts": counts,
        "missing_allowlist_intents": sorted(
            set(intents) - {e["target"]["intent"] for e in converted}
        ),
        "exact_duplicate_groups": exact_duplicates,
        "cross_split_exact_duplicate_groups": [
            group for group in exact_duplicates if len({e["split"] for e in group}) > 1
        ],
        "near_duplicate_audit": "pending",
        "manual_review": "pending",
        "test_targets_materialized": False,
    }
    output.mkdir(parents=True, exist_ok=False)
    for split in SPLITS:
        examples = [example for example in converted if example["split"] == split]
        write_json(
            output / f"{split}.manifest.json",
            {**provenance, "split": split, "example_ids": [e["id"] for e in examples]},
        )
        if split != "test":
            write_jsonl(output / f"{split}.jsonl", examples)
    review_pool = [e for e in converted if e["split"] != "test"]
    review = random.Random(20261020).sample(
        review_pool, min(config["manual_review_minimum"], len(review_pool))
    )
    write_jsonl(output / "manual-review.train-dev.jsonl", review)
    write_json(output / "audit.json", report)
    return report
