"""Verify the pinned stack with a tiny LoRA optimizer step on the allocated GPU."""

import argparse
import json
import sys
import tomllib
from importlib.metadata import version
from pathlib import Path


def verify(qlora: bool) -> dict:
    import torch
    from accelerate import Accelerator
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen3Config, Qwen3ForCausalLM, Trainer

    profile_dir = Path(__file__).resolve().parent
    profile = tomllib.loads((profile_dir / "pyproject.toml").read_text())["tool"]["sllm"]["cuda"]
    versions = {name: version(name) for name in ("torch", "accelerate", "transformers", "peft")}
    print(
        json.dumps(
            {
                "versions": versions,
                "torch_cuda": torch.version.cuda,
                "python": sys.executable,
                "torch_path": torch.__file__,
            },
            indent=2,
        ),
        flush=True,
    )
    if versions["torch"] != profile["torch_version"]:
        raise RuntimeError(
            f"Wrong Torch build: {versions['torch']}; expected {profile['torch_version']}"
        )
    if torch.version.cuda != profile["runtime_version"]:
        raise RuntimeError(f"Wrong CUDA runtime in Torch: {torch.version.cuda}")
    if not torch.cuda.is_available():
        raise RuntimeError("Pinned Torch cannot initialize CUDA; inspect the NVIDIA driver/runtime")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expose one GPU for this single-GPU training profile")
    precision = "bf16" if torch.cuda.is_bf16_supported(including_emulation=False) else "fp16"
    accelerator = Accelerator(mixed_precision=precision)
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=64,
    )
    model = get_peft_model(
        Qwen3ForCausalLM(config),
        LoraConfig(
            r=2,
            lora_alpha=4,
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
        ),
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad], lr=1e-4
    )
    model, optimizer = accelerator.prepare(model, optimizer)
    tokens = torch.randint(0, 32, (1, 8), device=accelerator.device)
    with accelerator.autocast():
        loss = model(input_ids=tokens, labels=tokens).loss
    accelerator.backward(loss)
    optimizer.step()
    if not torch.isfinite(loss):
        raise RuntimeError("GPU training smoke test produced a nonfinite loss")
    if qlora:
        import bitsandbytes as bnb

        dtype = torch.bfloat16 if precision == "bf16" else torch.float16
        layer = bnb.nn.Linear4bit(64, 64, quant_type="nf4", compute_dtype=dtype).to("cuda")
        inputs = torch.randn(2, 64, device="cuda", dtype=dtype, requires_grad=True)
        layer(inputs).float().square().mean().backward()
    torch.cuda.synchronize()
    report = {
        "status": "passed",
        "gpu": torch.cuda.get_device_name(0),
        "capability": torch.cuda.get_device_capability(0),
        "precision": precision,
        "versions": versions,
        "cuda_runtime": torch.version.cuda,
        "accelerate_lora_backward_optimizer": True,
        "nf4_backward": qlora,
        "trainer_import": Trainer.__name__,
    }
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qlora", action="store_true")
    verify(parser.parse_args().qlora)
