"""Create a separate training interpreter without replacing Colab's CUDA/PyTorch."""

import argparse
import json
import subprocess
import sys
import tomllib
from pathlib import Path


def bootstrap(environment: Path, qlora: bool) -> None:
    if not (3, 11) <= sys.version_info[:2] < (3, 13):
        raise RuntimeError("Use Python 3.11/3.12; Colab Runtime Version 2026.07 uses Python 3.12")
    root = Path(__file__).resolve().parents[2]
    probe = (
        "import json,torch; from importlib.metadata import version; "
        "print(json.dumps({'version':version('torch'),'file':torch.__file__,"
        "'cuda':torch.cuda.is_available(),'gpu':torch.cuda.get_device_name(0) "
        "if torch.cuda.is_available() else None}))"
    )
    before = json.loads(subprocess.check_output([sys.executable, "-c", probe], text=True))
    if not before["cuda"]:
        raise RuntimeError("Select a GPU runtime in Colab before setup; CUDA is unavailable")
    if not environment.exists():
        subprocess.run(
            [sys.executable, "-m", "venv", "--system-site-packages", str(environment)], check=True
        )
    python = str(environment / "bin/python")
    constraint = environment / "colab-torch-constraint.txt"
    constraint.write_text(f"torch=={before['version']}\n")
    subprocess.run(
        [
            python,
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
            "--constraint",
            str(constraint),
            "-r",
            str(root / "training/colab/requirements.txt"),
        ],
        check=True,
    )
    subprocess.run([python, "-m", "pip", "install", "--no-deps", "-e", str(root)], check=True)
    if qlora:
        lock = tomllib.loads((root / "uv.lock").read_text())
        version = next(
            package["version"] for package in lock["package"] if package["name"] == "bitsandbytes"
        )
        subprocess.run(
            [
                python,
                "-m",
                "pip",
                "install",
                "--no-cache-dir",
                "--no-deps",
                f"bitsandbytes=={version}",
            ],
            check=True,
        )
    after = json.loads(subprocess.check_output([python, "-c", probe], text=True))
    if before != after:
        raise RuntimeError(f"PyTorch/GPU environment changed unexpectedly: {before} -> {after}")
    subprocess.run(
        [
            python,
            "-c",
            "import transformers,peft,accelerate; "
            "print('Training imports OK:',transformers.__version__,peft.__version__,"
            "accelerate.__version__)",
        ],
        check=True,
    )
    print(json.dumps({"training_python": python, "inherited_torch": after}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", type=Path, default=Path("/content/sllm-venv"))
    parser.add_argument("--qlora", action="store_true")
    args = parser.parse_args()
    bootstrap(args.environment, args.qlora)
