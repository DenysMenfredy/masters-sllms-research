"""Optional real training/merge/resume check with tiny random weights, no downloads."""

import argparse
import json
import subprocess

import pytest
import yaml

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
transformers = pytest.importorskip("transformers", minversion="5.3")


def test_tiny_cpu_training_resume_and_merge(tmp_path):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

    from sllm_research.finetune import run_training
    from sllm_research.merge_adapter import merge_run

    torch.set_num_threads(1)
    model_dir = tmp_path / "tiny-base"
    vocabulary = {
        "<unk>": 0,
        "<pad>": 1,
        "<s>": 2,
        "</s>": 3,
        "<assistant>": 4,
        "wake": 5,
        "at": 6,
        "five": 7,
        "am": 8,
    }
    backend = Tokenizer(models.WordLevel(vocabulary, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
    )
    tokenizer.chat_template = (
        "{% for message in messages %}<s>{{ message['content'] }}</s>{% endfor %}"
        "{% if add_generation_prompt %}<assistant>{% endif %}"
    )
    tokenizer.save_pretrained(model_dir)
    Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=len(vocabulary),
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            eos_token_id=3,
            bos_token_id=2,
            pad_token_id=1,
            max_position_embeddings=2048,
        )
    ).save_pretrained(model_dir)
    root = tmp_path / "project"
    (root / "configs/models").mkdir(parents=True)
    (root / "configs/models/candidates.yaml").write_text(
        yaml.safe_dump(
            {
                "candidates": {
                    "tiny": {
                        "repository": str(model_dir),
                        "revision": "local",
                        "tokenizer_revision": "local",
                    }
                }
            }
        )
    )
    (root / "configs/experiment.yaml").write_text(yaml.safe_dump({"gates": {}}))
    subprocess.run(["git", "init", str(root)], check=True, capture_output=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "--allow-empty",
            "-m",
            "test",
        ],
        check=True,
        capture_output=True,
    )
    config = root / "smoke.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "mode": "smoke",
                "max_steps": 2,
                "save_steps": 1,
                "train_limit": 2,
                "dev_limit": 2,
                "rank": 2,
                "alpha": 4,
                "max_new_tokens": 4,
                "max_sequence_length": 1024,
                "gradient_accumulation_steps": 1,
                "epochs": 1,
            }
        )
    )
    for split in ("train", "dev"):
        rows = [
            {
                "id": f"{split}:{index}",
                "source_id": f"{split}:{index}",
                "split": split,
                "locale": "en-US",
                "utterance": "wake at five am",
                "target": {"intent": "alarm_set", "slots": [{"type": "time", "value": "five am"}]},
            }
            for index in range(2)
        ]
        (root / f"{split}.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    args = argparse.Namespace(
        root=root,
        config=config,
        candidate="tiny",
        seed=17,
        train=root / "train.jsonl",
        dev=root / "dev.jsonl",
        output=root / "run",
        cpu_smoke=True,
        resume=False,
    )
    summary = run_training(args)
    assert summary["examples"] == 2
    assert (args.output / "checkpoint-2/checkpoint.complete.json").exists()
    # Simulate interruption during finalization, after the last optimizer checkpoint.
    (args.output / "complete.json").rename(args.output / "first-complete.synthetic.json")
    args.resume = True
    run_training(args)
    merged = merge_run(args.output, root / "merged", args.dev)
    assert merged["canary_raw_output_agreement"]
    assert (root / "merged/model.safetensors").is_file()
