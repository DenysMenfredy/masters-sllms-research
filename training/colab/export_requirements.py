"""Export hash-locked base and QLoRA stacks from the dedicated CUDA profile."""

import subprocess
from pathlib import Path

profile_dir = Path(__file__).resolve().parent
for filename, extras in (
    ("requirements.txt", []),
    ("requirements-qlora.txt", ["--extra", "qlora"]),
):
    exported = subprocess.run(
        [
            "uv",
            "export",
            "--project",
            str(profile_dir),
            "--locked",
            "--no-dev",
            "--no-emit-project",
            "--no-annotate",
            "--no-header",
            *extras,
        ],
        text=True,
        capture_output=True,
        check=True,
    ).stdout
    header = (
        "# Generated from training/colab/uv.lock; do not edit individual pins.\n"
        "--index-url https://pypi.org/simple\n"
        "--extra-index-url https://download.pytorch.org/whl/cu126\n\n"
    )
    (profile_dir / filename).write_text(header + exported)
