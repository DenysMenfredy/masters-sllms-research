"""Read-only workstation diagnostics; unavailable measurements stay null."""

import platform
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import yaml


def environment_report(root: Path) -> dict:
    packages = {}
    for name in ("pydantic", "PyYAML", "pytest", "ruff", "torch", "transformers", "peft"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    tools = {name: shutil.which(name) for name in ("uv", "adb", "cmake", "ninja", "nvidia-smi")}
    cuda_driver = {"status": "unverified", "details": None}
    if tools["nvidia-smi"]:
        try:
            probe = subprocess.run(
                [
                    tools["nvidia-smi"],
                    "--query-gpu=name,driver_version,memory.total",
                    "--format=csv",
                ],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            cuda_driver = {
                "status": "available" if probe.returncode == 0 else "unavailable",
                "details": (probe.stdout + probe.stderr).strip(),
            }
        except (OSError, subprocess.TimeoutExpired) as error:
            cuda_driver = {"status": "unavailable", "details": str(error)}
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        code_revision = revision.stdout.strip() if revision.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        code_revision = None
    experiment = yaml.safe_load((root / "configs/experiment.yaml").read_text())
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "code_revision": code_revision,
        "packages": packages,
        "tools": tools,
        "cuda_driver": cuda_driver,
        "android_device_status": "unverified; run android/inspect_device.sh with an attached phone",
        "primary_model": experiment["primary_model"],
        "pending_requirement_fields": [
            key for key, value in experiment["requirements"].items() if value is None
        ],
        "gates": experiment["gates"],
    }
