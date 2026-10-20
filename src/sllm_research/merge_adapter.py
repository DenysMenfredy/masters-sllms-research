"""Reload the pinned high-precision base, merge an adapter, and save artifact provenance."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from sllm_research.finetune import generate_predictions, load_base
from sllm_research.io import read_jsonl, sha256_file, write_json, write_jsonl
from sllm_research.training_data import select_panel


def merge_run(run: Path, output: Path, dev: Path) -> dict:
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer

    metadata = json.loads((run / "run.json").read_text())
    complete = json.loads((run / "complete.json").read_text())
    adapter = run / complete["artifact_directory"] / "adapter"
    if sha256_file(adapter / "adapter_model.safetensors") != complete["adapter_sha256"]:
        raise ValueError("Adapter checksum mismatch")
    if sha256_file(dev) != metadata["dev_sha256"]:
        raise ValueError("Canary source must be the same development file as the training run")
    if output.exists():
        raise FileExistsError("Choose a new merged output directory")
    dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else (torch.float16 if torch.cuda.is_available() else torch.float32)
    )
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(adapter)
    # Always reload original unquantized weights, even when training used QLoRA/NF4.
    model = PeftModel.from_pretrained(
        load_base(metadata["candidate"], dtype, "lora", device), adapter
    )
    all_rows = read_jsonl(dev)
    if any(row.get("split") != "dev" or row.get("locale") != "en-US" for row in all_rows):
        raise ValueError("Canaries must use the official English development split")
    canaries = select_panel([row for row in all_rows if row["id"] in metadata["dev_ids"]], 8)
    before = generate_predictions(
        model, tokenizer, canaries, metadata["vocabulary"], metadata["recipe"]["max_new_tokens"]
    )
    merged = model.merge_and_unload(safe_merge=True)
    after = generate_predictions(
        merged, tokenizer, canaries, metadata["vocabulary"], metadata["recipe"]["max_new_tokens"]
    )
    output.mkdir(parents=True)
    merged.config.use_cache = True
    merged.save_pretrained(output, safe_serialization=True, max_shard_size="2GB")
    tokenizer.save_pretrained(output)
    write_jsonl(output / "canary.before-merge.jsonl", before)
    write_jsonl(output / "canary.after-merge.jsonl", after)
    manifest = {
        "created_utc": datetime.now(UTC).isoformat(),
        "source_run": str(run),
        "base": metadata["candidate"],
        "training_method": metadata["recipe"]["method"],
        "seed": metadata["seed"],
        "dtype": str(dtype),
        "format": "huggingface_safetensors",
        "deployment_quantization": None,
        "canary_raw_output_agreement": all(
            a["prediction"] == b["prediction"] for a, b in zip(before, after, strict=True)
        ),
        "files_sha256": {
            str(path.relative_to(output)): sha256_file(path)
            for path in sorted(output.rglob("*"))
            if path.is_file()
        },
    }
    write_json(output / "artifact.json", manifest)
    if not manifest["canary_raw_output_agreement"]:
        raise RuntimeError("Merge changed canary outputs; inspect saved outputs before GGUF export")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(merge_run(args.run, args.output, args.dev), indent=2))


if __name__ == "__main__":
    main()
