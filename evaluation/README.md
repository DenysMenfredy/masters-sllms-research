# Evaluation entry points

```bash
sllm schema
sllm score --gold data/processed/productivity-v0/dev.jsonl \
  --predictions /path/to/dev-predictions.jsonl \
  --output results/derived/unique-run-id.score.json
```

Gold rows require `id` and canonical `target`; converted rows also carry the
utterance and official split metadata. Predictions are JSONL envelopes with `id`
and a **raw response string** in `prediction`, for example:

```json
{"id":"en-US:123","prediction":"{\"intent\":\"alarm_set\",\"slots\":[]}"}
```

Include malformed responses verbatim. Absent predictions stay in the gold
denominator and score zero. Empty prediction files are valid and score all examples
as missing. Unknown/duplicate IDs and malformed envelopes fail. The scorer produces
per-example raw/parsed outputs, validity/error status, and semantic scores alongside
aggregates. It does not claim confidence intervals or device latency measurements.
See `docs/protocol.md` for exact metric definitions.
