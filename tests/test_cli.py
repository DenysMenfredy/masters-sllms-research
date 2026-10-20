import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_cli_score_keeps_invalid_and_missing_outputs():
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "sllm_research.cli",
            "score",
            "--gold",
            str(ROOT / "tests/fixtures/gold.jsonl"),
            "--predictions",
            str(ROOT / "tests/fixtures/predictions.jsonl"),
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    result = json.loads(completed.stdout)
    assert result["summary"]["examples"] == 3
    assert result["summary"]["joint_semantic_exact_match"] == 1 / 3
    assert result["summary"]["invalid_or_missing_predictions"] == 2
