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

## Environment and precision

`training/colab/bootstrap.py` creates `/content/sllm-venv` with access to preinstalled
system packages. It installs exact user-space pins from
`training/colab/requirements.txt`, generated from `uv.lock`. It constrains Torch to
its original installed distribution version and verifies Torch's version/path,
CUDA availability and GPU name afterward. CUDA/NVIDIA/Triton packages are omitted
from this export; they come from Colab's image. A compatible Datasets version is
pinned because Trainer can import an inherited installation even with JSONL input.

The notebook kernel stays unchanged. All training cells call the separate Python
interpreter, so no kernel restart is required. The Colab stack intentionally differs
from full workstation `uv sync --extra training` in Torch/CUDA packages. Every run
saves `pip-freeze.txt`, CUDA version, GPU name, dtype, recipe, model/code revisions,
template hash and data hashes. Record the chosen runtime version in the diary.
Use the same runtime image and exact repository commit for final runs and resume.
The notebook prints its resolved commit; `run.json` also retains it. A moving `main`
is a first-run convenience, not a reproducibility pin.

LoRA is the default. Training uses BF16 when supported by the allocated GPU,
otherwise FP16. CPU infrastructure checks use FP32. QLoRA is an explicit NF4 memory
fallback with double quantization. Set `USE_QLORA=True` for setup and training and
use a new run ID. NF4 training is separate from deployment Q4_K_M/Q8_0 precision.

After an intentional dependency update, regenerate the Colab export:

```bash
UV_CACHE_DIR="$PWD/.cache/uv" uv lock
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
| Import/dependency conflict | Start a fresh runtime and use the bootstrap |
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
