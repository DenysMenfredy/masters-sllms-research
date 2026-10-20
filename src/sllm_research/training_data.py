"""Dependency-light prompt, masking, data-boundary, and training recipe checks."""

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from sllm_research.io import read_jsonl, sha256_file
from sllm_research.semantics import SemanticAction

PROMPT_VERSION = "productivity-json-v1"
SYSTEM_PROMPT = (
    "Parse the English productivity request. Return only a JSON object with intent and slots. "
    "Each slot has type and value. Copy slot values verbatim from the request, preserve "
    "repeated slots, and use an empty slots list when none are present. Do not invent arguments."
)


class TrainingRecipe(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["smoke", "trial", "final"]
    method: Literal["lora", "qlora"] = "lora"
    rank: int = Field(default=8, gt=0)
    alpha: int = Field(default=16, gt=0)
    dropout: float = Field(default=0.05, ge=0, lt=1)
    learning_rate: float = Field(default=1e-4, gt=0)
    epochs: int = Field(default=3, ge=1, le=3)
    micro_batch_size: int = Field(default=1, gt=0)
    gradient_accumulation_steps: int = Field(default=16, gt=0)
    max_sequence_length: int = Field(default=512, gt=0)
    max_new_tokens: int = Field(default=128, gt=0)
    max_steps: int = -1
    train_limit: int | None = Field(default=None, gt=0)
    dev_limit: int | None = Field(default=None, gt=0)
    save_steps: int = Field(default=100, gt=0)
    gradient_checkpointing: bool = True
    target_modules: list[str] = ["q_proj", "k_proj", "v_proj", "o_proj"]

    @model_validator(mode="after")
    def bounded_run(self):
        if not self.target_modules or len(set(self.target_modules)) != len(self.target_modules):
            raise ValueError("target_modules must be nonempty and unique")
        if self.mode == "smoke":
            if not 1 <= self.max_steps <= 20 or self.train_limit is None or self.dev_limit is None:
                raise ValueError("Smoke runs require 1-20 steps and explicit train/dev limits")
        elif self.max_steps != -1 or self.train_limit is not None or self.dev_limit is not None:
            raise ValueError("Trial/final runs use all train/dev records and epoch-based selection")
        return self


def load_recipe(path: Path) -> TrainingRecipe:
    return TrainingRecipe.model_validate(yaml.safe_load(path.read_text()))


def validate_gates(recipe: TrainingRecipe, experiment: dict, candidate: str, seed: int) -> None:
    required = (
        []
        if recipe.mode == "smoke"
        else ["G1_verified_training_to_phone", "G2_dataset_and_protocol_frozen"]
    )
    if recipe.mode == "final":
        required.append("G3_training_recipe_frozen")
    missing = [gate for gate in required if not experiment["gates"].get(gate)]
    if missing:
        raise ValueError(f"Research gates are incomplete: {', '.join(missing)}")
    if recipe.mode != "smoke" and experiment["primary_model"] != candidate:
        raise ValueError("Trial/final candidate must equal the recorded primary_model")
    if recipe.mode == "final" and seed not in experiment["final_training_seeds"]:
        raise ValueError("Final training seed must be one of the predeclared final seeds")


def load_splits(train_path: Path, dev_path: Path) -> tuple[list[dict], list[dict]]:
    splits = []
    for path, expected in ((train_path, "train"), (dev_path, "dev")):
        rows = read_jsonl(path)
        if not rows:
            raise ValueError(f"Empty {expected} split")
        ids = set()
        for row in rows:
            if row.get("split") != expected or row.get("locale") != "en-US":
                raise ValueError(f"Expected only official en-US {expected} records in {path}")
            if row["id"] in ids:
                raise ValueError(f"Duplicate {expected} example ID: {row['id']}")
            ids.add(row["id"])
            if not isinstance(row.get("utterance"), str) or not row["utterance"]:
                raise ValueError("Every example needs a nonempty utterance")
            SemanticAction.model_validate(row["target"])
        splits.append(rows)
    train_ids = {row["id"] for row in splits[0]}
    train_source_ids = {row.get("source_id", row["id"]) for row in splits[0]}
    if any(
        row["id"] in train_ids or row.get("source_id", row["id"]) in train_source_ids
        for row in splits[1]
    ):
        raise ValueError("Train/dev source IDs overlap")
    return splits[0], splits[1]


def select_panel(rows: list[dict], limit: int | None, seed: int = 20261020) -> list[dict]:
    """Stable intent/slot-pattern round-robin panel shared across model/seed comparisons."""
    if limit is None or limit >= len(rows):
        return sorted(rows, key=lambda row: row["id"])
    groups = defaultdict(list)
    for row in sorted(rows, key=lambda row: row["id"]):
        key = (row["target"]["intent"], tuple(sorted(s["type"] for s in row["target"]["slots"])))
        groups[key].append(row)
    rng = random.Random(seed)
    keys = sorted(groups)
    rng.shuffle(keys)
    for group in groups.values():
        rng.shuffle(group)
    panel = []
    while len(panel) < limit:
        for key in keys:
            if groups[key] and len(panel) < limit:
                panel.append(groups[key].pop())
    return panel


def render_prompt(tokenizer, utterance: str, vocabulary: dict) -> str:
    system = SYSTEM_PROMPT + "\nAllowed vocabulary: " + json.dumps(vocabulary, sort_keys=True)
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": utterance}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def encode_example(tokenizer, row: dict, vocabulary: dict, max_length: int) -> tuple[dict, dict]:
    prompt = render_prompt(tokenizer, row["utterance"], vocabulary)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    target = SemanticAction.model_validate(row["target"]).model_dump_json()
    target_ids = tokenizer.encode(target, add_special_tokens=False)
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer has no EOS token")
    target_ids.append(tokenizer.eos_token_id)
    ids = prompt_ids + target_ids
    audit = {
        "id": row["id"],
        "prompt_tokens": len(prompt_ids),
        "response_tokens": len(target_ids),
        "total_tokens": len(ids),
        "truncated": False,
    }
    if len(ids) > max_length:
        raise ValueError(
            f"{row['id']} needs {len(ids)} tokens, exceeds {max_length}; "
            "increase max_sequence_length after the length audit; no truncation allowed"
        )
    return {
        "input_ids": ids,
        "attention_mask": [1] * len(ids),
        "labels": [-100] * len(prompt_ids) + target_ids,
    }, audit


def pad_features(features: list[dict], pad_token_id: int) -> dict:
    width = max(len(feature["input_ids"]) for feature in features)
    result = {key: [] for key in ("input_ids", "attention_mask", "labels")}
    for feature in features:
        padding = width - len(feature["input_ids"])
        result["input_ids"].append(feature["input_ids"] + [pad_token_id] * padding)
        result["attention_mask"].append(feature["attention_mask"] + [0] * padding)
        result["labels"].append(feature["labels"] + [-100] * padding)
    return result


def mark_checkpoint_complete(checkpoint: Path) -> None:
    """Write a completion marker only after model, optimizer, scheduler and RNG were saved."""
    required = {
        "trainer_state.json",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
        "adapter_model.safetensors",
    }
    if not all((checkpoint / name).is_file() for name in required):
        raise ValueError(f"Checkpoint lacks resume state: {checkpoint}")
    hashes = {
        path.name: sha256_file(path)
        for path in checkpoint.iterdir()
        if path.is_file() and not path.name.startswith("checkpoint.complete.json")
    }
    temporary = checkpoint / "checkpoint.complete.json.tmp"
    temporary.write_text(json.dumps({"files_sha256": hashes}, indent=2) + "\n")
    temporary.replace(checkpoint / "checkpoint.complete.json")


def latest_complete_checkpoint(output: Path) -> Path:
    candidates = sorted(
        (path for path in output.glob("checkpoint-*") if path.name.split("-")[-1].isdigit()),
        key=lambda path: int(path.name.split("-")[-1]),
        reverse=True,
    )
    for checkpoint in candidates:
        try:
            hashes = json.loads((checkpoint / "checkpoint.complete.json").read_text())[
                "files_sha256"
            ]
            if hashes and all(
                sha256_file(checkpoint / name) == checksum for name, checksum in hashes.items()
            ):
                return checkpoint
        except (OSError, ValueError, KeyError):
            continue
    raise ValueError(
        "No complete checkpoint to resume; choose a new run if interruption preceded saving"
    )
