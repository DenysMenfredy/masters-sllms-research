# Draft protocol v0.1

Source: research plan v1.1, pages 2-6 and 9-10. **Not frozen.** Requirements and
the primary model remain undecided. All configurations are starting points; their
presence is not evidence that G1 or G2 has passed.

## Semantic representation

```json
{"intent":"alarm_set","slots":[{"type":"time","value":"five am"}]}
```

`intent` and `slots` are required; every slot requires `type` and `value`. Extra
fields, duplicate JSON keys, invalid types, prose, Markdown fences, and truncated
JSON are rejected. Empty slots are `[]`. The current structural schema does not
enforce the study's final intent/slot vocabulary; add frozen vocabulary constraints
to both generation and validation at G2.

Converter `massive-annot-v1` removes official `[type : value]` annotation wrappers,
checks exact utterance reconstruction, and preserves annotation occurrence order,
duplicate slots, spelling, case, and whitespace. Unexpected syntax fails with its
source ID for review. Nothing fills missing arguments. IDs are `locale:source_id`;
the split is independent metadata, so a repeated source ID across splits fails.

Scorer `verbatim-v1` uses exact strings with no case folding or date normalization.
Joint semantic exact match requires valid output, the correct intent, and the same
complete multiset of `(type, value)` slots. JSON key and slot ordering do not affect
the score, while repeated-slot multiplicity does. Malformed/missing predictions
score zero joint/intent correctness and contribute missing gold slots to micro-F1.
Malformed prediction envelopes, duplicate IDs, or unknown IDs are input errors.

Micro-F1 is `2 TP / (2 TP + FP + FN)`, using exact slot type/value multisets. A corpus
with no gold or predicted slots has slot F1 1 by vacuous-match convention; interpret
it alongside validity and joint accuracy. Extra-or-invented slot rate is unmatched
predicted slots / all parsed predicted slots (0 when none are predicted). This is
not an independent factual hallucination measurement. Missing/invalid predictions
are counted separately and always remain in the denominator of primary accuracy.

## Data boundary

Use MASSIVE 1.1 en-US official train/dev/test partitions and the draft productivity
allowlist. Produce counts and split manifests before freezing. The initial audit
materializes train/dev targets only, plus test IDs and aggregate counts. Do not use
test examples or test error analysis to tune the allowlist, prompts, or recipe.
Exact duplicate detection does not replace near-duplicate or annotation review.
Any remediation needs a documented decision; never silently delete final test cases.

## Experiment controls

Pilot both pinned Qwen candidates on a shared stratified development panel of about
200 examples. Explicitly disable reasoning where supported and record the actual
tokenizer/template. Fix the primary model after its LoRA/merge/GGUF-to-phone path
works. Verify Qwen3.5 text-only training and conversion independently.

Core deployment matrix: untuned Q4_K_M/Q8_0 plus Q4_K_M/Q8_0 for each adapted seed
17, 42, and 73: eight edge artifacts. Keep high-precision workstation references.
Use identical prompts, output schema, decoding, context limits, and constraints for
matched untuned/adapted comparisons. Do not fabricate training gains.

Performance: a fixed roughly 200-request panel in three independent sessions per
artifact; randomize artifact order within sessions. Sustained tests last 20 minutes
for untuned and adapted Q4/Q8, using preselected adapted seed 17, in three sessions.
Separate initialization, first request, and warmed requests. Measure request latency
with a monotonic timer through tokenization and completion; retain RSS/PSS metric
names, units, and sampling intervals. Keep all failures and session boundaries.

Predeclare thresholds, timeouts, memory sampling, cooling, and device/power controls
before final tests. Their `null` config values are open decisions, not zero limits.
If there are no product thresholds, report the measured quality/latency/memory
frontier. A paired intent-stratified 10,000-replicate bootstrap and seed/session
variation are planned; inference, bootstrap, aggregation, and device benchmarks are
not implemented by this scaffold.

## Request logging contract to implement

Retain run/session ID, UTC timestamp, code revision and dirty state, dataset release
and example ID, model/tokenizer/template/config hashes, seed and precision, device
build, requested backend and actual offload, threads/context/batch/KV settings,
prompt/output token counts, raw and parsed output, all semantic scores and validity,
monotonic timing fields with units, RSS/PSS and system available memory with units,
thermal sensor identity/status and units, battery/charging state, cache state,
timeout and errors. Store unavailable observations as `null` with a reason.
Benchmark logging and schema enforcement are subsequent implementation work.
