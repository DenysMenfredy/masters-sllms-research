# Environment setup

Run commands from the project root. Python 3.11 is selected in `.python-version`;
the package supports Python 3.11 and 3.12. The tested local interpreter is 3.11.5.
All direct and transitive dependencies, including optional extras, are recorded in
`uv.lock`. Keep the lockfile under version control with the source.

```bash
make setup
source .venv/bin/activate
make check
make smoke
sllm doctor --output results/raw/environment-unique-run-id.json
```

Output files are created exclusively and are never silently overwritten. Choose a
new name for each diagnostic run. `doctor` uses metadata and a bounded `nvidia-smi`
probe; it does not import Torch, load models, contact a service, or assert that a
phone is connected. Environment timestamps are UTC; study dates use America/Belem.

The initial host has CMake and uv. The NVIDIA driver probe fails; ADB and Ninja are
absent and no NDK was configured. CPU data and scoring development works independently.
Use [the Colab training workflow](colab.md) for GPU work; no local GPU upgrade is
needed. Its bootstrap installs the dedicated Torch 2.14.1 CUDA 12.6 profile and
training libraries in an isolated interpreter. Workstation training extras
remain optional. Real-model GPU and Qwen3.5 feasibility still require smoke pilots.

Optional cache settings:

```bash
cp .env.example .env
set -a
source .env
set +a
```

Keep credentials out of command output, configuration files, and raw logs. Model
downloads should use the exact revisions in `configs/models/candidates.yaml`,
with weights under `artifacts/` or the ignored Hugging Face cache. Retain model
cards/licenses and compute tokenizer/template/artifact hashes after download.

For MASSIVE, use the 1.1 archive linked from the official repository, retain its
license and original archive checksum, and use `1.1/data/en-US.jsonl`. The audit
also records the exact locale file hash; a release label alone does not authenticate
downloaded content. No dataset or model is downloaded by setup or tests.

Git is initialized with the `main` branch and the requested GitHub `origin`:
`https://github.com/DenysMenfredy/masters-sllms-research.git`. The initial environment
report predates Git initialization and retains `code_revision: null`. Generate a new
report before each experiment so it records the actual checked-out commit.
