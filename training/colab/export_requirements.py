"""Export user-space training pins while leaving CUDA/PyTorch to the Colab image."""

import re
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[2]
exported = subprocess.run(
    [
        "uv",
        "export",
        "--locked",
        "--extra",
        "training",
        "--no-dev",
        "--no-emit-project",
        "--no-hashes",
        "--no-annotate",
    ],
    cwd=root,
    text=True,
    capture_output=True,
    check=True,
).stdout
lines = []
for line in exported.splitlines():
    name = re.split(r"[= ;]", line)[0].lower()
    if line.startswith("#") or not line.strip():
        continue
    if name == "torch" or name == "triton" or name.startswith(("nvidia-", "cuda-")):
        continue
    lines.append(line)
header = (
    "# Generated from uv.lock by training/colab/export_requirements.py.\n"
    "# PyTorch, Triton and CUDA libraries come from Colab; bootstrap constrains Torch\n"
    "# to the exact preinstalled build. Record the full resulting pip freeze per run.\n"
)
(root / "training/colab/requirements.txt").write_text(header + "\n".join(lines) + "\n")
