"""Standalone JSON CLI for the CampusOps Agent."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any

from campus_ops_agent.service import create_service
from campus_ops_agent.variants import VARIANTS


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the CampusOps showcase agent")
    parser.add_argument("message", nargs="?", help="Natural-language campus request")
    parser.add_argument("--input-json", help="Structured request JSON")
    parser.add_argument("--variant", choices=VARIANTS, default="fixed")
    parser.add_argument("--planner", choices=("deterministic", "deepseek"), default="deterministic")
    return parser


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    if args.input_json:
        request = json.loads(args.input_json)
    elif args.message:
        request = args.message
    elif not sys.stdin.isatty():
        request = json.loads(sys.stdin.read())
    else:
        raise SystemExit("provide a message, --input-json, or JSON on stdin")
    service = create_service(args.variant, planner_mode=args.planner)
    return await service.invoke(request)


def main() -> None:
    result = asyncio.run(_run(_parser().parse_args()))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
