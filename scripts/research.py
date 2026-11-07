"""Offline operational research CLI. No model training is implied by status/review."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.research_workflow import Workflow, WorkflowError, write_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("status", "next"):
        sub.add_parser(command)
    for command in ("validate-plan", "run", "verify", "review", "close"):
        child = sub.add_parser(command)
        child.add_argument("stage")
        if command in ("verify", "review", "close"):
            child.add_argument("--attempt")
        if command == "verify":
            child.add_argument("--historical", action="store_true", help="Verify frozen evidence without requiring live sources to match")
        if command == "review":
            choice = child.add_mutually_exclusive_group(required=True)
            choice.add_argument("--prepare", type=Path, help="Write a bound draft for the reviewer to score")
            choice.add_argument("--review", type=Path, help="Freeze a scored, evidence-bound review")
            child.add_argument("--report", help="Repository-relative report path (required for --prepare)")
    args = parser.parse_args(argv)
    try:
        workflow = Workflow(args.root)
        if args.command == "status":
            result = workflow.status()
        elif args.command == "next":
            result = workflow.next_stage()
        elif args.command == "validate-plan":
            result = workflow.validate_plan(args.stage)
        elif args.command == "run":
            result = workflow.run(args.stage)
        elif args.command == "verify":
            result = workflow.verify(args.stage, args.attempt, live_sources=not args.historical)
        elif args.command == "review":
            if args.prepare:
                if not args.report:
                    raise WorkflowError("review --prepare requires --report")
                result = workflow.review_template(args.stage, args.report, args.attempt)
                write_json(args.prepare, result, exclusive=True)
            else:
                result = workflow.review(args.stage, args.review, args.attempt)
        else:
            result = workflow.close(args.stage, args.attempt)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except (WorkflowError, OSError, KeyError, TypeError) as exc:
        print(json.dumps({"error": str(exc), "scientific_validity": "not_assessed"}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
