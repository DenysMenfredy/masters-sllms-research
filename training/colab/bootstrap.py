"""Install a fully isolated, locked CUDA training stack and verify actual GPU training."""

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import tomllib
from pathlib import Path

BOOTSTRAP_PIP_VERSION = "26.2"


def create_environment(environment: Path, child_env: dict) -> str:
    # Colab's distro Python can have venv but no working ensurepip. Build the
    # environment without seeding, then let the notebook's pip target it directly.
    subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(environment)],
        env=child_env,
        check=True,
    )
    configuration = (environment / "pyvenv.cfg").read_text().lower()
    if "include-system-site-packages = false" not in configuration:
        raise RuntimeError("Training interpreter still inherits system packages; use a fresh path")
    python = str(environment / "bin/python")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "--python",
            python,
            "install",
            "--no-cache-dir",
            "--only-binary=:all:",
            f"pip=={BOOTSTRAP_PIP_VERSION}",
        ],
        env=child_env,
        check=True,
    )
    return python


def version_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.strip().split("."))


def inspect_driver(minimum: str) -> list[dict]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
            timeout=20,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        details = getattr(error, "stderr", None) or str(error)
        raise RuntimeError(
            "No usable NVIDIA GPU/driver was detected. In Colab choose Runtime > Change "
            "runtime type > GPU and reconnect. Installing Torch cannot allocate a GPU.\n" + details
        ) from error
    devices = []
    for row in csv.reader(result.stdout.splitlines()):
        if not row:
            continue
        name, driver = (value.strip() for value in row)
        if version_tuple(driver) < version_tuple(minimum):
            raise RuntimeError(
                f"Driver {driver} is older than this CUDA 12.6 profile's minimum {minimum}. "
                "Select a newer Colab GPU runtime; do not replace Colab's host driver with pip."
            )
        devices.append({"gpu": name, "driver": driver})
    if not devices:
        raise RuntimeError("No GPU allocated. Select a Colab GPU runtime and reconnect.")
    return devices


def bootstrap(environment: Path, qlora: bool) -> None:
    if not (3, 11) <= sys.version_info[:2] < (3, 13):
        raise RuntimeError("Use Python 3.11/3.12; Colab Runtime Version 2026.07 uses Python 3.12")
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("This Colab CUDA profile targets Linux x86_64")
    profile_dir = Path(__file__).resolve().parent
    root = profile_dir.parents[1]
    profile = tomllib.loads((profile_dir / "pyproject.toml").read_text())["tool"]["sllm"]["cuda"]
    drivers = inspect_driver(profile["minimum_linux_driver"])
    print(json.dumps({"profile": profile, "driver_probe": drivers}, indent=2), flush=True)
    requirements = profile_dir / ("requirements-qlora.txt" if qlora else "requirements.txt")
    child_env = os.environ.copy()
    child_env.pop("PYTHONPATH", None)
    child_env["PYTHONNOUSERSITE"] = "1"
    python = create_environment(environment, child_env)
    subprocess.run(
        [
            python,
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
            "--require-hashes",
            "-r",
            str(requirements),
        ],
        env=child_env,
        check=True,
    )
    subprocess.run(
        [python, "-m", "pip", "install", "--no-deps", "-e", str(root)], env=child_env, check=True
    )
    subprocess.run([python, "-m", "pip", "check"], env=child_env, check=True)
    probe = [python, str(profile_dir / "verify_cuda.py")]
    if qlora:
        probe.append("--qlora")
    subprocess.run(probe, env=child_env, check=True)
    metadata = {
        "profile": profile,
        "driver_probe": drivers,
        "qlora": qlora,
        "requirements_sha256": hashlib.sha256(requirements.read_bytes()).hexdigest(),
        "bootstrap_pip_version": BOOTSTRAP_PIP_VERSION,
    }
    (environment / "sllm-environment.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"CUDA training environment ready: {python}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", type=Path, default=Path("/content/sllm-cu126-venv"))
    parser.add_argument("--qlora", action="store_true")
    args = parser.parse_args()
    bootstrap(args.environment, args.qlora)
