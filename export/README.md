# Adapter merge and deployment export

The high-precision merge stage is implemented as `sllm-merge` or
`python -m sllm_research.merge_adapter`; the Colab notebook includes an optional cell:

```bash
sllm-merge --run /path/to/completed/run \
  --dev /path/to/the/original/dev.jsonl --output /path/to/new/merged-directory
```

It verifies adapter/data hashes, reloads the pinned unquantized base, merges the
selected adapter, saves safetensors/tokenizer and provenance, and compares fixed
development canaries before/after merging. Divergence preserves evidence and fails
for investigation. Reloaded-GGUF/phone canary checks remain separate work.

Reload the original base at appropriate high precision before merging a LoRA or
QLoRA adapter. Evaluate the merged reference, convert that same checkpoint to GGUF,
then quantize to Q8_0 and Q4_K_M. GGUF conversion/quantization automation is pending.
Export untuned weights with the same pinned tools.
Save commands, dependency lock, exact revisions, licenses, tokenizer/template and
artifact checksums, sizes, canary IDs/outputs, and all failures. Verify actual phone
execution for both candidates at G1. Eight core edge artifacts are required after
final training: two untuned plus two for each of three adapted seeds.
