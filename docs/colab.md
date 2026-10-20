# Training and fine-tuning on Google Colab

[Open the notebook in Colab](https://colab.research.google.com/github/DenysMenfredy/masters-sllms-research/blob/main/notebooks/colab_finetune.ipynb)

Use Colab for LoRA/QLoRA training and adapter merging. Local data/scoring development
needs no GPU; deployment measurements still run on the Realme phone. Training uses
the network for public downloads. The offline requirement applies to deployment.

## First run

1. Open the link above, or upload `notebooks/colab_finetune.ipynb` through Colab's
   **File > Upload notebook**.
2. Choose **Runtime > Change runtime type > GPU** and Python 3.11/3.12. The documented
   **2026.07** image has Python 3.12.13 and PyTorch 2.11.0; select it in **Runtime
   Version** if the default is incompatible. GPU availability varies. See the
   [runtime guide](https://research.google.com/colaboratory/runtime-version-faq.html)
   and [resource FAQ](https://research.google.com/colaboratory/faq.html).
3. Keep candidate `qwen3_0_6b`, seed `17`, and `configs/training/colab_smoke.yaml` for
   the first attempt. Choose a new descriptive `RUN_ID`.
4. Run setup. CUDA and training-library import checks must succeed.
5. Mount Drive when Colab requests your authorization. Default storage is
   `MyDrive/masters-sllms-research`. Leave Drive enabled for resumable work.
6. Run the data cell to download official MASSIVE 1.1 and audit the candidate en-US
   subset. It saves train/dev, manifests, counts, license/checksum provenance, and a
   manual-review sample. It withholds test text/targets. The allowlist is not frozen.
7. Run training and inspect the outputs. Default smoke: **64 train examples,
   16 dev examples, two optimizer steps**. This checks the pipeline, not model gains.
8. After inspection, enable `DO_MERGE` for the optional merge/canary cell. Retain the
   merged checkpoint for the separate GGUF/phone pilot. Colab success alone is not G1.

Run the notebook interactively. Colab can terminate sessions and does not guarantee
GPU type or session length. No paid plan or compute purchase is made by the notebook.
For a private repository, opening the notebook and cloning code require your own
authenticated GitHub access. Never embed tokens in a notebook or committed URL.

## Upgrade an older notebook after a setup failure

Open the updated GitHub-backed notebook; saved Colab copies keep their old cells.
Choose a GPU runtime and rerun the fetch and setup cells. The new interpreter is
`/content/sllm-cu126-venv/bin/python`. Setup probes `nvidia-smi` first and imports
Torch from its isolated training environment afterward. If adapting an existing
notebook, replace the old setup cell with this after fetching the updated code:

```python
PYTHON = Path("/content/sllm-cu126-venv/bin/python")
bootstrap = [sys.executable, "training/colab/bootstrap.py", "--environment", PYTHON.parent.parent]
if USE_QLORA:
    bootstrap.append("--qlora")
run(bootstrap)
```

Use a new run ID if an earlier training run used different library versions.
The dataset and checkpoints in Drive are preserved across environment setup.

If an older fetch cell stops with **Repository has local changes**, run this once
to save tracked edits and untracked files, then rerun the fetch and setup cells:

```python
run(
    [
        "git",
        "-c",
        "user.name=SLLM Colab",
        "-c",
        "user.email=colab@localhost",
        "stash",
        "push",
        "--include-untracked",
        "--message",
        "sllm-colab-before-update",
    ]
)
```

The updated notebook fetches the requested revision first, shows changed filenames,
and automatically saves a named Git stash before checkout. It prints the stash
hash and inspection/restoration commands. Ignored data and caches stay in place.
If fetching or stashing fails, checkout does not proceed. Local edits saved in the
stash are excluded from the updated run until deliberately restored.

Inspect backups with `run(["git", "stash", "list"])` and
`run(["git", "stash", "show", "--include-untracked", "--stat", "stash@{0}"])`.
Restore selected changes after fetching and before training. For exact recovery,
check out the backup's original commit (`git checkout --detach <backup-hash>^1`)
and use `git stash apply --index <backup-hash>`; this also restores staged state.
For a resumed run, the code revision and tracked edits must match the saved run.
Stashes live in this VM's Git repository; export wanted edits before deleting the
runtime. Drive remains the persistent home for datasets, checkpoints and results.

If setup failed with `ensurepip ... returned non-zero exit status 1`, fetch the
repair and rerun the existing bootstrap command in the same notebook:

```python
run(["git", "fetch", "origin", "main"])
run(["git", "checkout", "--detach", "FETCH_HEAD"])
run(bootstrap)
```

The repaired setup reuses the partially created environment. It needs no system
package installation or kernel restart.

## Environment and precision

`training/colab/pyproject.toml` and its dedicated `uv.lock` define one resolved
Linux x86_64 / Python 3.11-3.12 CUDA profile. Its exact pins are:

| Component | Version |
| --- | --- |
| PyTorch | `2.14.1+cu126` |
| Bundled CUDA runtime | `12.6.77` (`torch.version.cuda` reports `12.6`) |
| Accelerate | `1.15.0` |
| Transformers | `5.20.2` |
| PEFT | `0.21.2` |
| Datasets | `4.8.5` |
| Optional bitsandbytes | `0.50.2` |

The CUDA-specific Torch wheel is listed in the
[official PyTorch index](https://download.pytorch.org/whl/cu126/torch/). The explicit
`+cu126` pin prevents the resolver from selecting a CPU or CUDA 13 build. The host
must expose an NVIDIA GPU with Linux driver **560.28.03 or newer**, the conservative
CUDA 12.6 GA requirement in NVIDIA's
[release notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html).
Driver selection belongs to the Colab runtime; pip does not install a host driver.
The CUDA version displayed by `nvidia-smi` may be newer than the wheel's runtime
because NVIDIA supports backward-compatible drivers.

`bootstrap.py` creates a venv without system-package inheritance using
`venv --without-pip`. This bypasses Colab images with unavailable or failing
`ensurepip`. The notebook's pip (22.3 or newer) uses its
[`--python` option](https://pip.pypa.io/en/stable/topics/python-option/) to install
`pip==26.2` into the training interpreter. Subsequent installs use that interpreter's
pip. Setup installs the full hash-locked requirements, including Torch/CUDA
libraries, and runs `pip check`.
This removes ambient Torch/Accelerate/vision/audio/TensorFlow package combinations
from the training interpreter. The notebook kernel stays unchanged. All training
commands use the separate interpreter, so no kernel restart is required.

`verify_cuda.py` prints versions and locations, checks the exact Torch build/CUDA
runtime, and executes a tiny Qwen3 LoRA forward/backward/optimizer step through
Accelerate with the GPU's supported mixed precision. QLoRA setup also executes an
NF4 backward pass. This checks imports and actual kernels before downloading a
candidate model. Package resolution alone does not establish GPU feasibility.

The dedicated lock is separate from the workstation lock, whose default Torch may
use a different CUDA build. Every run saves `pip-freeze.txt`, CUDA version, GPU name,
dtype, recipe, model/code revisions, template and data hashes. Setup also saves its
profile/checksum in the venv's `sllm-environment.json`. Record the runtime image in
the diary; use the same image and exact code commit for final runs and resume.
The notebook prints its resolved commit and saves it in `run.json`.

LoRA is the default. Training uses BF16 when natively supported by the allocated GPU,
otherwise FP16. CPU infrastructure checks use FP32. QLoRA is an explicit NF4 memory
fallback with double quantization. Set `USE_QLORA=True` for setup and training and
use a new run ID. NF4 training is separate from deployment Q4_K_M/Q8_0 precision.

After an intentional change to the dedicated Colab profile, regenerate both the
base and QLoRA hash-locked exports:

```bash
UV_CACHE_DIR="$PWD/.cache/uv" uv lock --project training/colab
UV_CACHE_DIR="$PWD/.cache/uv" .venv/bin/python training/colab/export_requirements.py
```

## Data, masking, and semantic validation

The trainer accepts only en-US official `train` and `dev` records, rejects duplicate
or overlapping source IDs, and derives the vocabulary from training labels only.
Review at least 100 train/dev examples and complete near-duplicate/license/coverage
audits before G2. Annotation round-trip failures stop conversion for review;
do not silently filter them or manufacture gold arguments.

The shared prompt uses the tokenizer's chat template with `enable_thinking=False`,
canonical JSON and the allowed vocabulary. The training sequence is the inference
prompt followed by target JSON and EOS. Prompt/padding labels are `-100`; response
tokens and EOS contribute to loss. Both train/dev lengths are checked. Overlong
examples fail with the ID and required length; none are silently truncated/dropped.
Length audits retain ID, prompt/response/total tokens and `truncated: false`.

Smoke panels are deterministic intent/slot-pattern round-robin samples shared
across candidates and seeds. Trial/final recipes use full train/dev subsets.
Seed controls adapter initialization, dropout and data ordering; bitwise
reproducibility across different GPUs is not claimed.

Validation generates from **prompts alone**, without the gold response. Greedy
decoding, reasoning policy, prompt and output budget match untuned/adapted runs.
Both currently use unconstrained JSON generation; any constraints must be frozen
and validated before core comparisons. Raw invalid responses remain in scoring,
and `hit_output_limit` is logged.

A strictly better development joint semantic exact match saves an adapter snapshot;
ties retain the earlier snapshot. Smoke evaluation runs at the configured step
interval; full runs evaluate after each epoch. Selection uses neither training loss
nor final test accuracy. LoRA selects at its training precision; QLoRA selects at
NF4. The chosen adapter is reloaded into the unquantized base for saved adapted dev
diagnostics, matched to the untuned high-precision reference. These are development
diagnostics, not held-out/Q4/Q8 evidence.

## Storage, artifacts, and reconnects

Model/download caches stay on `/content`; Drive holds canonical data, provenance,
checkpoints and results. The original MASSIVE archive stays on the VM; its checksum
and extracted license material are copied to persistent dataset provenance. Retain
the archive separately if required. See Google's
[Drive I/O guidance](https://research.google.com/colaboratory/faq.html).

```text
MyDrive/masters-sllms-research/
  datasets/productivity-v0/           Train/dev, manifests, review, source provenance
  runs/<RUN_ID>/
    run.json                         Recipe, revisions, hashes, modules, IDs, hardware
    pip-freeze.txt                   Actual training environment
    chat_template.jinja              Recorded template
    length-audit.{train,dev}.jsonl    Length/masking provenance
    untuned.dev.predictions.jsonl    Raw untuned development responses
    untuned.dev.scores.json           Untuned per-example and aggregate scores
    checkpoint-<step>/               Adapter, optimizer, scheduler, RNG, Trainer state
      checkpoint.complete.json       Checksums; written after checkpoint save finishes
    validation/step-<step>-<time>/    Predictions, scores, best adapter snapshots
    selection.json                  Best score and relative snapshot path
    final-<time>/                    Selected adapter and high-precision dev diagnostics
    complete.json                   Final artifact location/checksum
  merged/<RUN_ID>/                    Optional merged weights, tokenizer, canaries, hashes
```

Checkpoints save periodically and at epoch end. Resume chooses the latest
checksum-verified complete checkpoint, ignoring partial/corrupt newer saves.
Run identity must match code commit/tracked diff, candidate revision, recipe, seed,
train/dev hashes, vocabulary and prompt version. The template must also match.
Core training-library/Python/CUDA versions and training precision must match too;
a changed GPU cannot silently switch an FP16 run to BF16. Resume events record the
actual GPU and checkpoint used.
Finalization attempts use separate directories to preserve earlier partial logs.

After a disconnect:

1. Select the same runtime version and GPU class where possible.
2. Set `REPOSITORY_REF` to the exact saved `code_revision` from `run.json`.
3. Restore the original `RUN_ID`, candidate, seed, recipe and QLoRA flag.
4. Set `RESUME=True`; rerun setup/mount/data cells and training using the same data.

If no complete checkpoint was saved, start a new run ID. Completed runs refuse to
resume. Do not change the recipe of a resumed experiment. Use optimizer/RNG
checkpoints you created yourself, not arbitrary third-party checkpoint files.

## Trials and three final seeds

The full template is `configs/training/colab_full.yaml`: rank 8, learning rate 1e-4,
up to three epochs, micro-batch 1 and accumulation 16. Confirm sequence length and
target modules from smoke diagnostics; these are starting choices, not optima.
Exact module names and trainable parameter counts are saved. Missing/nonlinear
target modules and missing base weights stop a run.

Before a trial, complete G1/G2 with evidence and record the chosen candidate as
`primary_model` in `configs/experiment.yaml`. Commit the protocol/config changes.
Use the full template with `mode: trial` and at most two substantive development
trials. Pilot ranks 8 and 16 and select the final recipe using development results.

After G3, save a frozen recipe with `mode: final` and run **17, 42, 73** separately
with unique run IDs and the same recipe. The entry point enforces gate completion,
primary-model selection and final seed membership. Do not automate a three-seed
sweep before verifying the training-to-phone path. Record the Colab spend cap,
runtime/GPU, interruptions and recipe decisions.

## CLI and merge handoff

These are the same entry points the notebook calls:

```bash
sllm-train --config configs/training/colab_smoke.yaml \
  --candidate qwen3_0_6b --seed 17 \
  --train /path/to/productivity-v0/train.jsonl \
  --dev /path/to/productivity-v0/dev.jsonl \
  --output /persistent/path/runs/unique-run-id

# Add --resume to the identical training command to restore an interrupted run.
sllm-merge --run /persistent/path/runs/unique-run-id \
  --dev /path/to/productivity-v0/dev.jsonl \
  --output /persistent/path/merged/unique-run-id
```

Merge checks adapter/data hashes, reloads the original unquantized base even after
QLoRA, and compares eight fixed dev canaries before/after merging. It saves weights,
tokenizer, outputs and hashes. Raw canary divergence preserves the evidence and
fails for investigation. This is an in-memory precision check; reloaded-model/GGUF
canaries and phone execution are separate verification. Follow
[export](../export/README.md) and [Android](../android/README.md). Colab timings are
not phone latency/memory results.

## Troubleshooting and validation limits

| Symptom | Action |
| --- | --- |
| CUDA unavailable | Select a GPU runtime, reconnect and rerun setup |
| Unsupported Python | Select the documented compatible past runtime version |
| Legacy fetch cell reports local changes | Save a stash with the recovery cell above, then rerun fetch/setup; the updated notebook does this automatically |
| `venv` fails while invoking `ensurepip` | Fetch the repair and rerun `run(bootstrap)` as documented above; the partial environment is reused |
| Import/dependency conflict | Open the updated notebook and use the isolated CUDA bootstrap |
| Driver below 560.28.03 | Choose a newer Colab GPU runtime; package installation cannot update the host driver |
| Kernel Torch assertion fails in an old cell | Replace the setup cell as documented above and fetch the updated code |
| Output exists | Restore exactly with `RESUME=True`, or choose a new run ID |
| No complete checkpoint | Start a new run; interruption preceded a complete save |
| Out of memory | Reduce micro-batch and adjust accumulation to retain effective batch size; audit length; explicitly choose a new QLoRA run if needed |
| Length check fails | Review required length and revise the recorded recipe as a new run |
| Annotation mismatch | Review the source ID/converter; do not silently drop examples |
| Drive quota/I/O error | Preserve evidence and verify checkpoint completion before resuming |
| Qwen3.5 missing weights/modules | Preserve the error; stop that candidate's pilot rather than train missing random weights |

Local checks cover masking, boundaries, EOS/padding, gates, panels, checkpoints and
notebook syntax. A tiny random CPU model exercises training/resume/merge. There is
no usable local GPU: real Qwen weights, Colab CUDA, QLoRA, Qwen3.5 loading and phone
exports still require pilot validation. No adaptation benefit is claimed.
