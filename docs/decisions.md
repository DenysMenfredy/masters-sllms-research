# Decision register

## D001 - 20 October 2026 - foundation mirrors plan v1.1

Use the page-10 workflow directories and a `src/` Python package. Workflow folders
hold stage documentation and later entry points; reusable conversion/scoring code
lives in one installable package to avoid ad hoc `PYTHONPATH` changes.

## D002 - lightweight development environment

Use Python 3.11, uv, and a committed lockfile. Install core/dev dependencies first;
keep GPU training/QLoRA and analysis as optional locked extras. The current host's
NVIDIA driver fails, so a GPU stack or training feasibility cannot be certified here.
No model or dataset download is needed to check the infrastructure.

## D003 - exact values and multiplicity

Start with `verbatim-v1`: no string normalization; duplicate-preserving canonical
slots and multiset comparison. Fail unfamiliar annotations for review and count
invalid/missing model responses as failures. Freeze changes before final testing.

## D004 - open decisions remain open

Pin metadata revisions for both candidates without selecting a winner. Keep product
thresholds, sequence length, target modules, phone/runtime settings, and spend cap
unset. Declare their status explicitly; no invented performance or requirements.
The CPU build helper is prepared but execution is unverified.

## D005 - preselected sustained seed

Use adapted seed 17 for the planned sustained comparison, selected before training
or test inspection. This does not remove seeds 42 and 73 from core quality and
performance experiments.

## D006 - draft data handling

Initial audits produce train/dev records and test metadata only. Keep official
partitions, reject repeated source IDs, and report duplicate utterances without
automatic deletion. Near-duplicate and manual reviews remain mandatory at G2.
