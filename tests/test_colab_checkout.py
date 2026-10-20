import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks/colab_finetune.ipynb"


def git(repository, *arguments, check=True):
    return subprocess.run(
        [
            "git",
            "-c",
            "user.name=Checkout test",
            "-c",
            "user.email=checkout-test@localhost",
            "-C",
            str(repository),
            *arguments,
        ],
        check=check,
        capture_output=True,
        text=True,
    ).stdout.strip()


@pytest.fixture
def checkout(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", "/dev/null")
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    origin = tmp_path / "origin"
    origin.mkdir()
    git(origin, "init", "--initial-branch=main")
    (origin / ".gitignore").write_text("ignored/\n")
    (origin / "tracked.txt").write_text("original\n")
    (origin / "delete-me.txt").write_text("keep in upstream\n")
    git(origin, "add", ".")
    git(origin, "commit", "-m", "Initial files")
    return SimpleNamespace(origin=origin, repo=tmp_path / "checkout")


def fetch_cell(checkout, reference="main"):
    notebook = json.loads(NOTEBOOK.read_text())
    cell = next(cell for cell in notebook["cells"] if cell["id"] == "sllm-colab-03")
    source = cell["source"]
    source = "".join(source) if isinstance(source, list) else source

    def run(command):
        subprocess.run([str(value) for value in command], cwd=checkout.repo, check=True)

    exec(
        compile(source, str(NOTEBOOK), "exec"),
        {
            "REPO": checkout.repo,
            "REPOSITORY_URL": checkout.origin.as_uri(),
            "REPOSITORY_REF": reference,
            "subprocess": subprocess,
            "run": run,
        },
    )


def test_fetch_cell_clones_clean_revision_without_a_stash(checkout):
    fetch_cell(checkout)
    assert git(checkout.repo, "rev-parse", "HEAD") == git(checkout.origin, "rev-parse", "HEAD")
    assert git(checkout.repo, "status", "--porcelain") == ""
    assert git(checkout.repo, "stash", "list") == ""


def test_fetch_cell_preserves_edits_and_new_files_then_updates(checkout, capsys):
    fetch_cell(checkout)
    original = git(checkout.repo, "rev-parse", "HEAD")
    (checkout.repo / "tracked.txt").write_text("staged change\n")
    git(checkout.repo, "add", "tracked.txt")
    (checkout.repo / "tracked.txt").write_text("unstaged change\n")
    (checkout.repo / "delete-me.txt").unlink()
    new_file = checkout.repo / "notes with spaces.bin"
    new_file.write_bytes(b"\x00local notes\xff")
    ignored = checkout.repo / "ignored" / "data.txt"
    ignored.parent.mkdir()
    ignored.write_text("persistent ignored data\n")
    (checkout.origin / "tracked.txt").write_text("upstream repair\n")
    git(checkout.origin, "commit", "-am", "Repair upstream")

    fetch_cell(checkout)
    backup = git(checkout.repo, "rev-parse", "refs/stash")
    assert backup in capsys.readouterr().out
    assert git(checkout.repo, "rev-parse", f"{backup}^1") == original
    assert git(checkout.repo, "show", f"{backup}:tracked.txt") == "unstaged change"
    assert git(checkout.repo, "show", f"{backup}^2:tracked.txt") == "staged change"
    assert (checkout.repo / "tracked.txt").read_text() == "upstream repair\n"
    assert not new_file.exists()
    assert (checkout.repo / "delete-me.txt").exists()
    assert ignored.read_text() == "persistent ignored data\n"
    assert git(checkout.repo, "status", "--porcelain") == ""
    assert git(checkout.repo, "rev-parse", "HEAD") == git(checkout.origin, "rev-parse", "HEAD")

    fetch_cell(checkout)
    assert git(checkout.repo, "rev-parse", "refs/stash") == backup
    assert len(git(checkout.repo, "stash", "list").splitlines()) == 1

    git(checkout.repo, "checkout", "--detach", original)
    git(checkout.repo, "stash", "apply", "--index", backup)
    assert (checkout.repo / "tracked.txt").read_text() == "unstaged change\n"
    assert git(checkout.repo, "show", ":tracked.txt") == "staged change"
    assert not (checkout.repo / "delete-me.txt").exists()
    assert new_file.read_bytes() == b"\x00local notes\xff"
    assert ignored.read_text() == "persistent ignored data\n"


def test_failed_fetch_leaves_edits_in_place(checkout):
    fetch_cell(checkout)
    original = git(checkout.repo, "rev-parse", "HEAD")
    (checkout.repo / "tracked.txt").write_text("local change\n")
    with pytest.raises(subprocess.CalledProcessError):
        fetch_cell(checkout, "missing-ref")
    assert git(checkout.repo, "rev-parse", "HEAD") == original
    assert (checkout.repo / "tracked.txt").read_text() == "local change\n"
    assert git(checkout.repo, "stash", "list") == ""


def test_unresolved_merge_stops_checkout_and_retains_conflicts(checkout):
    fetch_cell(checkout)
    git(checkout.repo, "checkout", "-b", "local")
    (checkout.repo / "tracked.txt").write_text("local branch\n")
    git(checkout.repo, "commit", "-am", "Local change")
    git(checkout.repo, "checkout", "main")
    (checkout.repo / "tracked.txt").write_text("main branch\n")
    git(checkout.repo, "commit", "-am", "Conflicting change")
    git(checkout.repo, "merge", "local", check=False)
    original = git(checkout.repo, "rev-parse", "HEAD")
    conflicts = git(checkout.repo, "ls-files", "--unmerged")
    content = (checkout.repo / "tracked.txt").read_text()
    assert conflicts

    with pytest.raises(subprocess.CalledProcessError):
        fetch_cell(checkout)
    assert git(checkout.repo, "rev-parse", "HEAD") == original
    assert git(checkout.repo, "ls-files", "--unmerged") == conflicts
    assert (checkout.repo / "tracked.txt").read_text() == content
