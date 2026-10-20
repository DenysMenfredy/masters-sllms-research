# Local runner SDK - implementation pending

Planned lifecycle: inspect device, validate configuration with a recorded reason,
load, infer with timeout and telemetry, and unload. Record requested backend and
actual offload independently. CPU is required. A memory-based selector is optional
after the core evidence gate and requires measured profiles and hysteresis.

The current CLI only performs host diagnostics, local data audit, schema export,
and scoring. There is no model lifecycle implementation yet. Date normalization and
proposal mapping must be separate from benchmark semantics; no real calendar,
alarm, or list action should execute as a side effect of scoring.
