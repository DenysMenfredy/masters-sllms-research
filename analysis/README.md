# Analysis stage - implementation pending

Read immutable request logs; write regenerable tables/figures to `results/derived/`.
Implement paired intent-stratified bootstrap of adapted-minus-untuned quality, with
the same sampled examples across models/seeds (10,000 replicates, 95% intervals).
Report all three seeds and variation separately. Preserve independent session
boundaries in device summaries; failures are outcomes, not dropped rows.

Planned figures: quality versus latency, quality versus peak process memory,
adaptation gain by precision, and sustained latency over time. No analysis is
implemented or result fabricated by this scaffold.
