# Implementation source register

Checked 20 October 2026. The supplied PDF is the research design; upstream docs
inform compatibility and setup. Dependencies are pinned in `uv.lock`; runtime
commit and dataset file/archive hashes must be recorded after acquisition.

- [Research plan v1.1](../masters_research_plan.pdf), especially pages 4-6 and 10.
- [MASSIVE official repository](https://github.com/alexa/massive): release 1.1,
  official partition names, JSONL/annotation format, and archive link. Retain
  archive LICENSE and upstream NOTICE/THIRD-PARTY materials; redistribution review
  is pending. No data has been redistributed in this scaffold.
- [Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B): tokenizer/template,
  thinking switch, and declared Apache-2.0 license.
- [Qwen3.5-0.8B model card](https://huggingface.co/Qwen/Qwen3.5-0.8B) and
  [Transformers documentation](https://huggingface.co/docs/transformers/model_doc/qwen3_5):
  hybrid architecture and text-only feasibility considerations. The installed global
  Transformers 4.57.6 is not adopted as the project training environment.
- [llama.cpp Android instructions](https://github.com/ggml-org/llama.cpp/blob/master/docs/android.md):
  NDK arm64 cross-compilation and portable CPU build flags. Source revision remains open.
- [uv lock and sync](https://docs.astral.sh/uv/concepts/projects/sync/): lockfile and
  optional-extra installation behavior.

Candidate revision metadata was read from
`https://huggingface.co/api/models/Qwen/Qwen3-0.6B` and
`https://huggingface.co/api/models/Qwen/Qwen3.5-0.8B`; the returned commits are in
`configs/models/candidates.yaml`. Metadata pinning is not model execution validation.

Colab implementation references, checked on the same date:

- [Colab FAQ](https://research.google.com/colaboratory/faq.html): GPU/session
  variability, persistent storage, and Drive I/O behavior.
- [Colab runtime versions](https://research.google.com/colaboratory/runtime-version-faq.html):
  selecting a past image and the documented 2026.07 Python/Torch versions.
- [Transformers Trainer](https://huggingface.co/docs/transformers/main_classes/trainer):
  training/checkpoint lifecycle. API compatibility is checked against the locked version.
- [PEFT quantization](https://huggingface.co/docs/peft/developer_guides/quantization):
  NF4 preparation and adapter training. This is separate from GGUF quantization.
- [Locked Qwen3.5 implementation](https://github.com/huggingface/transformers/blob/v5.20.2/src/transformers/models/qwen3_5/modeling_qwen3_5.py):
  dedicated causal text model class. Actual pretrained loading/export requires the pilot.

CUDA profile repair references:

- [Official Torch CUDA 12.6 wheels](https://download.pytorch.org/whl/cu126/torch/):
  `2.14.1+cu126` wheels for supported Python/OS combinations.
- [NVIDIA CUDA release notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html):
  CUDA 12.6 GA Linux driver requirement 560.28.03.
- [NVIDIA compatibility guide](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html):
  driver/runtime compatibility, including newer drivers running older runtimes.
- [bitsandbytes installation](https://huggingface.co/docs/bitsandbytes/installation):
  CUDA 12.6 support and NF4 hardware requirements. GPU execution still needs a pilot.
- [Python 3.12 venv](https://docs.python.org/3.12/library/venv.html):
  `--without-pip` skips the default `ensurepip` invocation and existing directories
  are reused.
- [pip interpreter selection](https://pip.pypa.io/en/stable/topics/python-option/):
  pip 22.3+ can manage another interpreter, including a venv without pip installed.
- [Git stash](https://git-scm.com/docs/git-stash): saving tracked/untracked changes,
  inspecting a backup and restoring staged state with `apply --index`.
