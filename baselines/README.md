# Required baselines - implementation pending

Implement both a simple rule-based joint parser and a compact jointly trained
intent/slot encoder after coverage audit. Use train/dev data only for development.
Evaluate the same canonical semantics and complete frozen final test subset. An
intent-only classifier cannot replace the full-task encoder. Set a fixed encoder
deployment budget; if it cannot run on the phone, report quality and explicitly
non-comparable workstation timings. Record licenses, revisions, and artifact hashes.
