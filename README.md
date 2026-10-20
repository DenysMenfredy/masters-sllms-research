# Offline productivity SLM research

Implementation of the **Master Research Execution Plan v1.1**, 20 October to
14 December 2026. The source of truth is [the research plan](masters_research_plan.pdf).
The study evaluates domain-specific LoRA adaptation and Q4/Q8 deployment on a
Realme 7 Pro using English calendar, alarm, and list requests.

## Current implementation

The initial development environment, experiment configurations, canonical semantic
schema, official MASSIVE annotation converter, draft data audit, and semantic scorer
are implemented. Training, model export, inference, specialist baselines, and device
benchmarking are upcoming work. No model quality or device performance is claimed.
G1 through G4 remain open; installing packages does not establish model feasibility.

## Start here

Requires Python 3.11 and [uv](https://docs.astral.sh/uv/getting-started/installation/).
Run from this directory:

```bash
make setup
make check
make smoke
make doctor
source .venv/bin/activate
sllm --help
```

`make setup` creates an isolated `.venv` from `uv.lock`, including development tools.
It installs no model weights, dataset, PyTorch, Android SDK, or CUDA components.
`make smoke` uses explicitly synthetic fixtures; its result is an infrastructure
check, not research evidence. The example has one correct prediction, one malformed
prediction, and one missing prediction: joint exact match and validity are both 1/3.

Optional environments use the same lockfile:

```bash
# GPU workstation only, after checking CUDA/driver compatibility:
UV_CACHE_DIR="$PWD/.cache/uv" uv sync --locked --group dev --extra training
# QLoRA fallback, if needed:
UV_CACHE_DIR="$PWD/.cache/uv" uv sync --locked --group dev --extra training --extra qlora
# Statistical analysis and figures, when needed:
UV_CACHE_DIR="$PWD/.cache/uv" uv sync --locked --group dev --extra analysis
```

Extras are resolved and locked, but only the default development environment has
been installed and checked. `uv sync` installs the requested extras exactly; include
all desired `--extra` flags together to keep them installed. Do not use the global
Conda environment as the experiment environment. Review the optional settings in
`.env.example`; commands do not load `.env` automatically.

## Repository layout

```text
configs/                 Draft experiment, data, model, training, and device configs
src/sllm_research/        Working CLI, semantics, MASSIVE conversion, scoring, doctor
data/raw/                Original downloads (ignored)
data/processed/          Regenerable audits and converted examples (ignored)
data/manifests/          Reviewed, versioned split IDs, schemas, and provenance
baselines/               Rule parser and joint intent/slot encoder work
training/                LoRA implementation work
export/                  Merge, GGUF, quantization, and canary work
android/                 Device inspection and CPU cross-compilation helpers
sdk/                     Future model lifecycle and proposal interfaces
evaluation/              Benchmark protocols and scoring format
analysis/                Future paired statistics, plots, and result tables
results/raw/             Immutable request and session logs (ignored)
results/derived/         Regenerable summaries (ignored)
artifacts/               Model artifacts and adapters (ignored)
external/                Pinned third-party checkouts (ignored)
docs/                    Setup, protocol, decisions, experiment diary, milestones
paper/                   Manuscript and thesis outlines
tests/                   Semantic correctness, conversion, split, and CLI checks
```

See [setup](docs/setup.md), [the draft protocol](docs/protocol.md),
[implementation milestones](docs/implementation.md), and
[the experiment diary](docs/experiments/2026-10-20.md).

## First data command

Obtain MASSIVE 1.1 from the [official repository](https://github.com/alexa/massive)
and retain its license and original archive. Supply the unmodified `en-US.jsonl`:

```bash
sllm audit-massive \
  --input data/raw/1.1/data/en-US.jsonl \
  --config configs/data/massive.yaml \
  --output data/processed/productivity-v0
```

The output directory must be new. The command keeps official partitions and source
IDs, writes source/config checksums, counts retained intents and slot combinations,
checks duplicate IDs and exact duplicate utterances, and produces a train/dev manual
review sample. It writes test IDs and aggregate counts, **not test text or targets**.
Near-duplicate review, at least 100 manual reviews, license review, and coverage
decisions remain required before G2. The candidate allowlist is not frozen.

## Reproducibility rules

- Select the primary model only after a verified LoRA-to-phone pilot (G1).
- Freeze schema, prompts, decoding, dataset, and requirements before final testing (G2).
- Preserve duplicate slots, missing arguments, malformed predictions, and failures.
- Record exact revisions, templates, hashes, units, and session controls; unknown is `null`.
- Keep training quantization separate from deployment Q4_K_M and Q8_0 quantization.
- Keep semantic parsing separate from date resolution and real action execution.
- Compare untuned and adapted artifacts through the same export and decoding path.

No project license has been selected. Third-party data, models, and runtime retain
their own licenses; see [the source register](docs/sources.md) before redistribution.
