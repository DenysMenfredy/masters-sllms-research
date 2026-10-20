import importlib.util
import subprocess
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

PROFILE = Path(__file__).resolve().parents[1] / "training/colab"
SPEC = importlib.util.spec_from_file_location("colab_bootstrap", PROFILE / "bootstrap.py")
bootstrap = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bootstrap)


def test_no_allocated_gpu_fails_before_any_package_install(monkeypatch):
    def no_driver(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi is absent")

    monkeypatch.setattr(bootstrap.subprocess, "run", no_driver)
    with pytest.raises(RuntimeError, match="Installing Torch cannot allocate a GPU"):
        bootstrap.inspect_driver("560.28.03")


def test_empty_gpu_inventory_is_not_treated_as_available(monkeypatch):
    monkeypatch.setattr(bootstrap.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=""))
    with pytest.raises(RuntimeError, match="No GPU allocated"):
        bootstrap.inspect_driver("560.28.03")


def test_old_driver_reports_the_profile_requirement(monkeypatch):
    monkeypatch.setattr(
        bootstrap.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(stdout="Tesla T4, 535.104.05\n"),
    )
    with pytest.raises(RuntimeError, match="minimum 560.28.03"):
        bootstrap.inspect_driver("560.28.03")


@pytest.mark.parametrize("driver", ["560.28.03", "580.65.06", "590.10.01"])
def test_driver_may_be_newer_than_the_wheel_cuda_runtime(monkeypatch, driver):
    monkeypatch.setattr(
        bootstrap.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=f"Tesla T4, {driver}\n")
    )
    assert bootstrap.inspect_driver("560.28.03")[0]["driver"] == driver


@pytest.mark.parametrize("qlora", [False, True])
def test_isolated_install_and_compute_probe_do_not_depend_on_kernel_torch(
    tmp_path, monkeypatch, qlora
):
    environment = tmp_path / "environment"
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        if command[0] == "nvidia-smi":
            return SimpleNamespace(stdout="Tesla T4, 580.65.06\n")
        if command[1:3] == ["-m", "venv"]:
            environment.mkdir()
            (environment / "pyvenv.cfg").write_text("include-system-site-packages = false\n")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(bootstrap.subprocess, "run", fake_run)
    monkeypatch.setenv("PYTHONPATH", "/old/incompatible/site-packages")
    bootstrap.bootstrap(environment, qlora)
    install = next((command, kwargs) for command, kwargs in calls if "--require-hashes" in command)
    assert install[0][-1].endswith("requirements-qlora.txt" if qlora else "requirements.txt")
    assert "PYTHONPATH" not in install[1]["env"]
    assert "--system-site-packages" not in str(calls)
    commands = [command for command, _ in calls]
    assert any(command[-2:] == ["pip", "check"] for command in commands)
    assert any("verify_cuda.py" in str(command) for command in commands)
    assert (environment / "sllm-environment.json").exists()


def test_cuda_profile_exports_keep_exact_gpu_build_and_hashes():
    profile = tomllib.loads((PROFILE / "pyproject.toml").read_text())
    lock = tomllib.loads((PROFILE / "uv.lock").read_text())
    torch = next(package for package in lock["package"] if package["name"] == "torch")
    assert torch["version"] == profile["tool"]["sllm"]["cuda"]["torch_version"]
    assert torch["source"]["registry"] == "https://download.pytorch.org/whl/cu126"
    for filename in ("requirements.txt", "requirements-qlora.txt"):
        exported = (PROFILE / filename).read_text()
        assert "torch==2.14.1+cu126" in exported
        assert "--hash=sha256:" in exported
        assert "--extra-index-url https://download.pytorch.org/whl/cu126" in exported
