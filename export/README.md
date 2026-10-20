# Export stage - implementation pending

Reload the original base at appropriate high precision before merging a LoRA or
QLoRA adapter. Evaluate the merged reference, convert that same checkpoint to GGUF,
then quantize to Q8_0 and Q4_K_M. Export untuned weights with the same pinned tools.
Save commands, dependency lock, exact revisions, licenses, tokenizer/template and
artifact checksums, sizes, canary IDs/outputs, and all failures. Verify actual phone
execution for both candidates at G1. Eight core edge artifacts are required after
final training: two untuned plus two for each of three adapted seeds.
