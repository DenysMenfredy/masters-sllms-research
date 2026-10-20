# Implementation milestones

## Completed foundation

- Directory layout matching plan page 10, with separate source and workflow folders.
- Python packaging, project CLI, isolated environment, dependency lock and dev tools.
- Draft configuration for data, model candidates, training, device, and experiment.
- Exact upstream candidate model revision metadata; primary model remains unselected.
- Semantic schema, annotation converter, draft split/coverage audit, and scorer.
- Synthetic fixtures and regression checks for the correctness-critical data path.
- Host diagnostics, raw ADB inventory helper, and pinned-source CPU build helper.
- Draft protocol, decision log, experiment diary, source register, and writing outlines.

## Colab training implementation

- GPU notebook with environment preflight, official data acquisition/audit and Drive storage.
- Pinned training packages while preserving the Colab image's Torch/CUDA distribution.
- Response-only LoRA/NF4 QLoRA entry point with deterministic panels and length checks.
- Generated dev scoring, semantic checkpoint selection and complete checkpoint resume.
- High-precision base reload, adapter merge, canary checks and artifact checksums.
- Local tiny-model CPU training/resume/merge integration check; real GPU pilots pending.

## G1: first verified training-to-phone path (20-26 October)

- [ ] Inventory the actual phone and fill firmware/ABI/device controls.
- [ ] Obtain compatible GPU access and agree on the training spend cap.
- [ ] Acquire MASSIVE with licenses/checksums; audit candidate coverage.
- [ ] Review train/dev ambiguity and freeze pilot development-panel IDs.
- [ ] Pin and build llama.cpp; preserve compiler/NDK flags and binary hashes.
- [ ] Run both candidate Q4/Q8 CPU pilots with templates/reasoning verified.
- [ ] Run small LoRA, merge, export, canary and phone execution for both candidates.
- [ ] Estimate experiment elapsed time with 30% contingency; select the primary model.
- [ ] Resolve novelty, contribution, requirements, and university milestones in review.

## G2: data and evaluation freeze (27 October-2 November)

- [ ] Finish at least 100 manual reviews, near-duplicate audit and license review.
- [ ] Freeze allowlist/vocabulary, manifests, gold conversion, prompts, and decoding.
- [ ] Implement full-task rule and joint intent/slot encoder baselines.
- [ ] Implement constrained runtime output and per-request telemetry/failure handling.
- [ ] Freeze application criteria, timeout/memory/cooldown controls, and analyses.
- [ ] Add controlled final-test materialization only after the freeze.

## G3-G8

Follow plan pages 7-8: two substantive training trials; three final seeds; merged
high-precision plus GGUF quantizations/canaries; artifact freeze; repeated quality,
performance and sustained collection; paired statistics and reproducible tables;
four core figures; thesis/manuscript; clean-environment reproduction. Optional GPU,
policy, energy, and PT-BR work cannot displace the core study.
