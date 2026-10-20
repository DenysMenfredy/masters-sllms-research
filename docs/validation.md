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
