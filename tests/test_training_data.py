import ast
import json
from pathlib import Path

import pytest

from sllm_research.training_data import (
    encode_example,
    latest_complete_checkpoint,
    load_recipe,
    load_splits,
    mark_checkpoint_complete,
    pad_features,
    select_panel,
    validate_gates,
)

ROOT = Path(__file__).resolve().parents[1]


class CharacterTokenizer:
    eos_token_id = 999

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] is False
        return "\n".join(message["content"] for message in messages) + "\nassistant:"

    def encode(self, text, **kwargs):
        return [ord(char) for char in text]


def row(identifier="train:1", split="train"):
    return {
        "id": identifier,
        "source_id": identifier,
        "split": split,
        "locale": "en-US",
        "utterance": "wake me at five am",
        "target": {"intent": "alarm_set", "slots": [{"type": "time", "value": "five am"}]},
    }


def test_loss_masks_prompt_but_preserves_response_and_eos():
    encoded, audit = encode_example(CharacterTokenizer(), row(), {}, 10000)
    boundary = audit["prompt_tokens"]
    assert all(label == -100 for label in encoded["labels"][:boundary])
    assert encoded["labels"][boundary:] == encoded["input_ids"][boundary:]
    assert encoded["labels"][-1] == CharacterTokenizer.eos_token_id
    assert not audit["truncated"]


def test_overlong_example_fails_instead_of_dropping_gold():
    with pytest.raises(ValueError, match="no truncation"):
        encode_example(CharacterTokenizer(), row(), {}, 10)


def test_padding_masks_only_padding_even_when_pad_is_eos():
    batch = pad_features(
        [
            {"input_ids": [1, 999], "attention_mask": [1, 1], "labels": [-100, 999]},
            {"input_ids": [999], "attention_mask": [1], "labels": [999]},
        ],
        999,
    )
    assert batch["labels"] == [[-100, 999], [999, -100]]
    assert batch["attention_mask"] == [[1, 1], [1, 0]]


@pytest.mark.parametrize("bad_split", ["test", "train"])
def test_training_loader_rejects_wrong_development_split(tmp_path, bad_split):
    train, dev = tmp_path / "train.jsonl", tmp_path / "dev.jsonl"
    train.write_text(json.dumps(row()))
    dev.write_text(json.dumps(row("dev:2", bad_split)))
    with pytest.raises(ValueError, match="official en-US dev"):
        load_splits(train, dev)


def test_training_loader_rejects_shared_source_ids(tmp_path):
    train, dev = tmp_path / "train.jsonl", tmp_path / "dev.jsonl"
    train.write_text(json.dumps(row()))
    dev.write_text(json.dumps({**row("dev:2", "dev"), "source_id": "train:1"}))
    with pytest.raises(ValueError, match="overlap"):
        load_splits(train, dev)


def test_development_panel_is_independent_of_input_order():
    rows = [row(str(index), "dev") for index in range(20)]
    assert select_panel(rows, 5) == select_panel(list(reversed(rows)), 5)


def test_full_training_cannot_bypass_research_gates():
    recipe = load_recipe(ROOT / "configs/training/colab_full.yaml")
    experiment = {"gates": {}, "primary_model": None, "final_training_seeds": [17, 42, 73]}
    with pytest.raises(ValueError, match="gates"):
        validate_gates(recipe, experiment, "qwen3_0_6b", 17)
    validate_gates(
        load_recipe(ROOT / "configs/training/colab_smoke.yaml"), experiment, "qwen3_0_6b", 17
    )


def test_resume_uses_latest_complete_uncorrupted_checkpoint(tmp_path):
    good = tmp_path / "checkpoint-1"
    good.mkdir()
    for name in (
        "trainer_state.json",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
        "adapter_model.safetensors",
    ):
        (good / name).write_text("synthetic resume state")
    mark_checkpoint_complete(good)
    incomplete = tmp_path / "checkpoint-2"
    incomplete.mkdir()
    assert latest_complete_checkpoint(tmp_path) == good
    (good / "optimizer.pt").write_text("corrupt")
    with pytest.raises(ValueError, match="No complete checkpoint"):
        latest_complete_checkpoint(tmp_path)


def test_notebook_cells_are_valid_python_with_no_saved_outputs():
    notebook = json.loads((ROOT / "notebooks/colab_finetune.ipynb").read_text())
    assert notebook["nbformat"] == 4
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse(cell["source"])
            assert cell["execution_count"] is None and not cell["outputs"]
