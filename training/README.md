# LoRA training

Start with [the Colab notebook](../notebooks/colab_finetune.ipynb) and
[the complete guide](../docs/colab.md). The implementation is
`src/sllm_research/finetune.py`, available as `sllm-train` or
`python -m sllm_research.finetune` with the training dependencies installed.

`configs/training/colab_smoke.yaml` is a bounded pipeline check; the full template
is `configs/training/colab_full.yaml`. Plan-level choices remain in `lora.yaml`.
Training masks prompts/padding, preserves response/EOS loss, fails overlong
examples, records modules/trainable counts, selects adapters by generated dev joint
semantic exact match, and saves optimizer/scheduler/RNG state with checksums.

Trials require G1/G2 and primary-model selection; final runs also require G3 and
seeds 17, 42, 73. Limit substantive development trials to two. QLoRA is an explicit
NF4 memory fallback, separate from Q4_K_M/Q8_0 deployment. Colab/real-model GPU
feasibility still needs pilot evidence.
