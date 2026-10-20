"""Single-GPU LoRA/QLoRA with generated semantic validation and resumable checkpoints."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import yaml

from sllm_research.evaluation import score_records
from sllm_research.io import sha256_file, write_json, write_jsonl
from sllm_research.training_data import (
    PROMPT_VERSION,
    encode_example,
    latest_complete_checkpoint,
    load_recipe,
    load_splits,
    mark_checkpoint_complete,
    pad_features,
    render_prompt,
    select_panel,
    validate_gates,
)


def load_base(candidate: dict, dtype, method: str, device: str):
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig, Qwen3_5ForCausalLM

    # Qwen3.5 has a dedicated text-only class; do not instantiate its vision encoder.
    cls = (
        Qwen3_5ForCausalLM
        if candidate["repository"] == "Qwen/Qwen3.5-0.8B"
        else AutoModelForCausalLM
    )
    options = {"revision": candidate["revision"], "dtype": dtype, "attn_implementation": "sdpa"}
    if method == "qlora":
        options.update(
            device_map={"": 0},
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=dtype,
            ),
        )
    model, loading = cls.from_pretrained(
        candidate["repository"], output_loading_info=True, **options
    )
    if loading.get("missing_keys") or loading.get("mismatched_keys"):
        raise ValueError(f"Base weights did not load completely: {loading}")
    if method != "qlora":
        model.to(torch.device(device))
    return model


def generate_predictions(
    model, tokenizer, rows: list[dict], vocabulary: dict, max_new_tokens: int
) -> list[dict]:
    import torch

    was_training = model.training
    model.eval()
    predictions = []
    try:
        with torch.inference_mode():
            for row in rows:
                prompt = render_prompt(tokenizer, row["utterance"], vocabulary)
                inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
                inputs = {key: value.to(model.device) for key, value in inputs.items()}
                generated = model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )[0, inputs["input_ids"].shape[1] :]
                predictions.append(
                    {
                        "id": row["id"],
                        "prediction": tokenizer.decode(
                            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
                        ),
                        "output_tokens": len(generated),
                        "hit_output_limit": len(generated) == max_new_tokens,
                    }
                )
    finally:
        model.train(was_training)
    return predictions


def run_training(args) -> dict:
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoTokenizer, Trainer, TrainerCallback, TrainingArguments, set_seed

    recipe = load_recipe(args.config)
    experiment = yaml.safe_load((args.root / "configs/experiment.yaml").read_text())
    validate_gates(recipe, experiment, args.candidate, args.seed)
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("This entry point supports one training process and one GPU")
    if not torch.cuda.is_available() and not args.cpu_smoke:
        raise ValueError("No CUDA GPU. In Colab choose Runtime > Change runtime type > GPU")
    if args.cpu_smoke and (recipe.mode != "smoke" or recipe.method == "qlora"):
        raise ValueError("CPU is supported only for non-quantized infrastructure smoke tests")
    if not args.cpu_smoke and torch.cuda.device_count() != 1:
        raise ValueError("Expose one GPU with CUDA_VISIBLE_DEVICES for this single-GPU recipe")
    candidates = yaml.safe_load((args.root / "configs/models/candidates.yaml").read_text())
    candidate = candidates["candidates"][args.candidate]
    train_rows, dev_rows = load_splits(args.train, args.dev)
    # Vocabulary is derived from training labels only, never validation or test labels.
    vocabulary = {
        "intents": sorted({row["target"]["intent"] for row in train_rows}),
        "slot_types": sorted({s["type"] for row in train_rows for s in row["target"]["slots"]}),
    }
    train_rows = select_panel(train_rows, recipe.train_limit)
    dev_rows = select_panel(dev_rows, recipe.dev_limit)
    candidate_identity = {**candidate, "candidate_key": args.candidate}
    identity = {
        "recipe": recipe.model_dump(),
        "candidate": candidate_identity,
        "seed": args.seed,
        "train_sha256": sha256_file(args.train),
        "dev_sha256": sha256_file(args.dev),
        "prompt_version": PROMPT_VERSION,
        "vocabulary": vocabulary,
        "runtime_versions": {
            name: version(name)
            for name in ("torch", "transformers", "peft", "accelerate")
            + (("bitsandbytes",) if recipe.method == "qlora" else ())
        },
        "python_version": sys.version.split()[0],
        "torch_cuda": torch.version.cuda,
    }
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=args.root, capture_output=True, text=True, check=True
    ).stdout.strip()
    diff = subprocess.run(
        ["git", "diff", "HEAD", "--binary"], cwd=args.root, capture_output=True, check=True
    ).stdout
    identity.update(code_revision=revision, tracked_diff_sha256=hashlib.sha256(diff).hexdigest())
    identity_hash = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    if args.resume:
        saved = json.loads((args.output / "run.json").read_text())
        if saved["identity_sha256"] != identity_hash:
            raise ValueError("Resume recipe/model/seed/data differs from the original run")
        if (args.output / "complete.json").exists():
            raise ValueError("Run is already complete; choose a new output for another run")
        resume_checkpoint = latest_complete_checkpoint(args.output)
    elif args.output.exists():
        raise FileExistsError("Output exists; use --resume or choose a new run directory")
    set_seed(args.seed)
    device = "cpu" if args.cpu_smoke else "cuda:0"
    dtype = (
        torch.float32
        if args.cpu_smoke
        else (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
    )
    if args.resume and saved["dtype"] != str(dtype):
        raise ValueError("Resume GPU would change training precision; use the original GPU class")
    tokenizer = AutoTokenizer.from_pretrained(
        candidate["repository"], revision=candidate["tokenizer_revision"]
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    features, audits = [], []
    for row in train_rows:
        feature, audit = encode_example(tokenizer, row, vocabulary, recipe.max_sequence_length)
        features.append(feature)
        audits.append(audit)
    # Audit validation lengths too; never truncate the prompt or gold response silently.
    dev_audits = [
        encode_example(tokenizer, row, vocabulary, recipe.max_sequence_length)[1]
        for row in dev_rows
    ]
    template_hash = hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()
    if args.resume and saved["chat_template_sha256"] != template_hash:
        raise ValueError("Tokenizer template differs from the original run")
    model = load_base(candidate, dtype, recipe.method, device)
    if recipe.method == "qlora":
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    module_names = [
        name
        for name, module in model.named_modules()
        if name.rsplit(".", 1)[-1] in recipe.target_modules and isinstance(module, torch.nn.Linear)
    ]
    if any(
        not any(name.endswith(f".{suffix}") for name in module_names)
        for suffix in recipe.target_modules
    ):
        raise ValueError("A requested LoRA target module is absent or is not linear")
    model = get_peft_model(
        model,
        LoraConfig(
            r=recipe.rank,
            lora_alpha=recipe.alpha,
            lora_dropout=recipe.dropout,
            target_modules=recipe.target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    model.config.use_cache = False
    args.output.mkdir(parents=True, exist_ok=True)
    if args.resume:
        write_json(
            args.output / f"resume-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%f')}.json",
            {
                "checkpoint": resume_checkpoint.name,
                "gpu": torch.cuda.get_device_name(0) if device != "cpu" else None,
                "dtype": str(dtype),
                "runtime_versions": identity["runtime_versions"],
            },
        )
    if not args.resume:
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=args.root,
                capture_output=True,
                text=True,
                check=True,
            ).stdout
        )
        write_json(
            args.output / "run.json",
            {
                **identity,
                "identity_sha256": identity_hash,
                "code_revision": revision,
                "code_dirty": dirty,
                "started_utc": datetime.now(UTC).isoformat(),
                "chat_template_sha256": template_hash,
                "dtype": str(dtype),
                "torch_cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0) if device != "cpu" else None,
                "target_module_names": module_names,
                "trainable_parameters": sum(
                    p.numel() for p in model.parameters() if p.requires_grad
                ),
                "total_parameters": sum(p.numel() for p in model.parameters()),
                "decoding": {
                    "do_sample": False,
                    "enable_thinking": False,
                    "max_new_tokens": recipe.max_new_tokens,
                    "json_constraints": False,
                },
                "train_ids": [row["id"] for row in train_rows],
                "dev_ids": [row["id"] for row in dev_rows],
            },
        )
        write_jsonl(args.output / "length-audit.train.jsonl", audits)
        write_jsonl(args.output / "length-audit.dev.jsonl", dev_audits)
        (args.output / "pip-freeze.txt").write_text(
            subprocess.run(
                [sys.executable, "-m", "pip", "freeze"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout
        )
        (args.output / "chat_template.jinja").write_text(tokenizer.chat_template)
        reference = (
            load_base(candidate, dtype, "lora", device) if recipe.method == "qlora" else model
        )
        baseline = generate_predictions(
            reference, tokenizer, dev_rows, vocabulary, recipe.max_new_tokens
        )
        write_jsonl(args.output / "untuned.dev.predictions.jsonl", baseline)
        write_json(args.output / "untuned.dev.scores.json", score_records(dev_rows, baseline))
        del reference
        if recipe.method == "qlora":
            torch.cuda.empty_cache()

    class SemanticTrainer(Trainer):
        def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
            predictions = generate_predictions(
                self.model, tokenizer, dev_rows, vocabulary, recipe.max_new_tokens
            )
            scores = score_records(dev_rows, predictions)
            metrics = {
                f"{metric_key_prefix}_{key}": value for key, value in scores["summary"].items()
            }
            evaluation_dir = (
                args.output
                / "validation"
                / (f"step-{self.state.global_step}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%f')}")
            )
            evaluation_dir.mkdir(parents=True)
            write_jsonl(evaluation_dir / "predictions.jsonl", predictions)
            write_json(evaluation_dir / "scores.json", scores)
            selection_path = args.output / "selection.json"
            previous = json.loads(selection_path.read_text()) if selection_path.exists() else None
            score = scores["summary"]["joint_semantic_exact_match"]
            if previous is None or score > previous["score"]:
                adapter_path = evaluation_dir / "adapter"
                self.model.save_pretrained(adapter_path, safe_serialization=True)
                selection = {
                    "adapter": str(adapter_path.relative_to(args.output)),
                    "score": score,
                    "global_step": self.state.global_step,
                    "metric": "dev_joint_semantic_exact_match",
                }
                temporary = selection_path.with_suffix(".json.tmp")
                temporary.write_text(json.dumps(selection, indent=2) + "\n")
                temporary.replace(selection_path)
            self.log(metrics)
            self.control = self.callback_handler.on_evaluate(
                self.args, self.state, self.control, metrics
            )
            return metrics

    class EpochCheckpoint(TrainerCallback):
        def on_epoch_end(self, args, state, control, **kwargs):
            control.should_save = True
            return control

        def on_save(self, args, state, control, **kwargs):
            mark_checkpoint_complete(Path(args.output_dir) / f"checkpoint-{state.global_step}")
            return control

    def collate(batch):
        return {
            key: torch.tensor(value, dtype=torch.long)
            for key, value in pad_features(batch, tokenizer.pad_token_id).items()
        }

    training_args = TrainingArguments(
        output_dir=str(args.output),
        num_train_epochs=recipe.epochs,
        max_steps=recipe.max_steps,
        learning_rate=recipe.learning_rate,
        per_device_train_batch_size=recipe.micro_batch_size,
        gradient_accumulation_steps=recipe.gradient_accumulation_steps,
        gradient_checkpointing=recipe.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        fp16=dtype == torch.float16,
        bf16=dtype == torch.bfloat16,
        use_cpu=args.cpu_smoke,
        eval_strategy="steps" if recipe.mode == "smoke" else "epoch",
        eval_steps=recipe.save_steps,
        save_strategy="steps",
        save_steps=recipe.save_steps,
        save_total_limit=2,
        save_only_model=False,
        load_best_model_at_end=False,
        metric_for_best_model="eval_joint_semantic_exact_match",
        greater_is_better=True,
        logging_steps=1,
        report_to="none",
        optim="adamw_torch",
        seed=args.seed,
        data_seed=args.seed,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        label_names=["labels"],
    )
    trainer = SemanticTrainer(
        model=model,
        args=training_args,
        train_dataset=features,
        eval_dataset=features[:1],
        data_collator=collate,
        processing_class=tokenizer,
        callbacks=[EpochCheckpoint()],
    )
    trainer.train(resume_from_checkpoint=str(resume_checkpoint) if args.resume else None)
    trainer.save_state()
    selection_path = args.output / "selection.json"
    if not selection_path.exists():
        raise RuntimeError("No validation adapter selected; check save/evaluation intervals")
    selection = json.loads(selection_path.read_text())
    best = args.output / selection["adapter"]
    from peft import PeftModel

    # Load selected adapter into a fresh high-precision base, including after QLoRA.
    del trainer, model
    __import__("gc").collect()
    if device != "cpu":
        torch.cuda.empty_cache()
    selected = PeftModel.from_pretrained(load_base(candidate, dtype, "lora", device), best)
    final_dir = args.output / f"final-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%f')}"
    final_dir.mkdir()
    selected.save_pretrained(final_dir / "adapter", safe_serialization=True)
    tokenizer.save_pretrained(final_dir / "adapter")
    final_predictions = generate_predictions(
        selected, tokenizer, dev_rows, vocabulary, recipe.max_new_tokens
    )
    write_jsonl(final_dir / "adapted.dev.predictions.jsonl", final_predictions)
    final_scores = score_records(dev_rows, final_predictions)
    write_json(final_dir / "adapted.dev.scores.json", final_scores)
    complete = {
        "best_adapter": selection["adapter"],
        "selection_metric": "dev_joint_semantic_exact_match",
        "best_metric": selection["score"],
        "artifact_directory": final_dir.name,
        "adapter_sha256": sha256_file(final_dir / "adapter/adapter_model.safetensors"),
        "completed_utc": datetime.now(UTC).isoformat(),
        "mode": recipe.mode,
    }
    write_json(args.output / "complete.json", complete)
    return final_scores["summary"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=Path("configs/training/colab_smoke.yaml"))
    parser.add_argument("--candidate", default="qwen3_0_6b")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cpu-smoke", action="store_true", help="Tiny infrastructure tests only")
    args = parser.parse_args()
    print(json.dumps(run_training(args), indent=2))


if __name__ == "__main__":
    main()
