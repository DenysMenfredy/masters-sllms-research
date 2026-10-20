# Training stage - implementation pending

Starting settings are in `configs/training/lora.yaml`. After G1 feasibility, build
response-only SFT with architecture-appropriate LoRA modules, seeded data ordering,
tokenizer length audit, explicit truncation counts, validation generation scored by
joint semantic exact match, and selected-checkpoint provenance. Inspect actual
masked labels and trainable parameter counts in a small smoke run. Three final
seeds are 17, 42, 73; limit substantive development trials to two. QLoRA is a
memory-driven fallback, not deployment quantization. No training entry point is
present yet; do not start full training before the phone export path is verified.
