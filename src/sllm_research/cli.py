"""Offline project entry points; no model or dataset downloads on import."""

import argparse
import json
from pathlib import Path

from sllm_research.data import audit_dataset
from sllm_research.doctor import environment_report
from sllm_research.evaluation import score_records
from sllm_research.io import read_jsonl, write_json
from sllm_research.semantics import SemanticAction


def main() -> None:
    parser = argparse.ArgumentParser(prog="sllm")
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor", help="Report workstation tools and unresolved gates")
    doctor.add_argument("--root", type=Path, default=Path.cwd())
    doctor.add_argument("--output", type=Path)
    schema = commands.add_parser("schema", help="Print the canonical semantic JSON schema")
    schema.add_argument("--output", type=Path)
    audit = commands.add_parser("audit-massive", help="Audit a local official en-US JSONL file")
    audit.add_argument("--input", type=Path, required=True)
    audit.add_argument("--config", type=Path, default=Path("configs/data/massive.yaml"))
    audit.add_argument("--output", type=Path, required=True)
    score = commands.add_parser("score", help="Score JSONL predictions against canonical gold")
    score.add_argument("--gold", type=Path, required=True)
    score.add_argument("--predictions", type=Path, required=True)
    score.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "doctor":
            result = environment_report(args.root)
        elif args.command == "schema":
            result = SemanticAction.model_json_schema()
        elif args.command == "audit-massive":
            result = audit_dataset(args.input, args.config, args.output)
        else:
            result = score_records(read_jsonl(args.gold), read_jsonl(args.predictions))
        if args.command != "audit-massive" and args.output:
            write_json(args.output, result)
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f"sllm: {error}\n")


if __name__ == "__main__":
    main()
