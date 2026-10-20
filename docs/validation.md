# Foundation validation - 20 October 2026

These checks validate infrastructure, not models or research hypotheses.

| Check | Outcome |
| --- | --- |
| `uv lock` | Resolved 99 package entries, including optional training/analysis stacks |
| `uv sync --locked --group dev` | Installed core/dev in local `.venv` with Python 3.11.5 |
| `make check` | Ruff lint/format pass; 23 tests pass |
| `make smoke` | Synthetic end-to-end scorer returns expected 1/3 joint accuracy and validity |
| `sllm schema` | Structural JSON schema saved under `data/manifests/` |
| `sllm doctor` | Saved `results/raw/environment-2026-10-20-initial.json` |
| `bash -n android/*.sh` | Both helper scripts pass syntax validation |

Coverage includes correct parses, wrong intents, absent/extra/repeated/empty slots,
verbatim matching, malformed/truncated JSON, duplicate keys, missing outputs,
prediction ID integrity, official split retention, source-ID leakage, annotation
round trips, test target withholding, checksums, and refusal to overwrite an audit.
No tests download data or weights, use CUDA, or require a phone.

Resolved optional training versions include Torch 2.14.1, Transformers 5.20.2,
PEFT 0.21.2, Accelerate 1.15.0 and Datasets 4.8.5. They are **not installed or
execution-validated** in the core development environment. Build backend Hatchling
1.32.4 is pinned separately in `pyproject.toml`. uv used for setup: 0.7.21.

The saved host probe confirms an unavailable NVIDIA driver and no ADB/Ninja on
PATH. No Android NDK was configured. Exact phone firmware, runtime commit, actual
offload, model/template hashes and code commit remain unverified. All experiment
gates and all deployment thresholds remain open in the draft configuration.

Reproduce environment consistency checks with an explicit interpreter because the
host also has an active Conda environment:

```bash
UV_CACHE_DIR="$PWD/.cache/uv" uv lock --check --offline
UV_CACHE_DIR="$PWD/.cache/uv" uv pip check --python .venv/bin/python
make check
make smoke
```

## Colab implementation checks

The default lightweight environment passes 33 tests; one optional ML integration
test skips because Torch is intentionally absent. Ruff checks include the notebook.
All code cells parse as Python and have no saved execution outputs.

A temporary environment with the exported user-space pins and the host's inherited
Torch 2.13.0+cu130 separately passes the integration test using a tiny randomly
initialized Qwen3 model and local tokenizer. It exercises two optimizer steps,
generated semantic validation, adapter selection, checkpoint state/checksums,
resume after interrupted finalization, high-precision reload, adapter merge, saved
weights and matching before/after canary outputs. No model/dataset download or
CUDA execution is involved. The CPU test is not a real-model quality experiment.

The integration check caught and fixed inherited old Datasets incompatibility and
the removed `TrainingArguments.save_safetensors` option in the pinned Transformers
version. Colab GPU, mixed precision, bitsandbytes QLoRA, real Qwen3.5 weights and
phone export still require pilot validation. That initial implementation inherited
Torch/CUDA from the runtime image; the isolated profile below supersedes that policy.

## CUDA profile repair checks

The dedicated `training/colab/uv.lock` resolves the complete Torch 2.14.1+cu126,
Accelerate 1.15.0, Transformers 5.20.2 and PEFT 0.21.2 stack, including the QLoRA
extra and CUDA 12.6 runtime libraries. Both base and QLoRA hash-locked pip exports
pass metadata-only installation dry runs with pip-compatible index matching across
the trusted PyPI/PyTorch sources. No GPU wheel is installed locally by these checks.

The lightweight suite now passes 42 tests; the optional ML integration test skips
without Torch. New regression checks cover absent GPUs, empty inventories, old and
newer compatible drivers, isolated package installation, removal of stale Python
paths, hash enforcement and the exact CUDA-specific Torch pin. Notebook code cells
parse and have no saved execution outputs.

The new bootstrap runs `pip check` in the installed Colab interpreter, followed by
a tiny actual LoRA optimizer step through Accelerate on its GPU. QLoRA additionally
checks NF4 backpropagation. Those GPU checks cannot run on this local host and must
pass in Colab before loading a candidate checkpoint. A concrete exception from the
original failed session was not supplied; the dependency/isolation repair does not
claim to reproduce that unknown traceback.
