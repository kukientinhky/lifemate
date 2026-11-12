"""Streaming LifeMate development pipeline: one ID at a time.

For every sample this performs, in order:
    ASR for latest/history user audio -> LLM tool planning -> ToolEnv -> JSONL row

The ASR model is loaded once, but each sample's result is written and flushed
before the next sample begins. This is the execution shape required by the
30-second-per-sample competition limit. OpenRouter use in this script is for
development only; final Docker inference must replace it with a local LLM.

Example:
    .venv/bin/python main/run_dev_pipeline.py --split public_test \
        --id public_0002 --model qwen/qwen3-235b-a22b-2507 \
        --output artifacts/public_0002_submission.jsonl
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import time
from typing import Any

from agent import DEFAULT_MODEL, catalog_by_name, infer, read_jsonl
from asr import PhoWhisperTranscriber
from openrouter_test import load_project_env


def add_transcripts(item: dict[str, Any], split_dir: Path, transcriber: PhoWhisperTranscriber) -> dict[str, Any]:
    """Return an in-memory copy enriched with current and historical ASR text."""
    enriched = copy.deepcopy(item)
    enriched["audio_text"] = transcriber.transcribe(split_dir / enriched["audio"])
    for turn in enriched.get("history", []):
        if turn.get("role") == "user" and turn.get("audio"):
            turn["audio_text"] = transcriber.transcribe(split_dir / turn["audio"])
    return enriched


def asr_row(item: dict[str, Any]) -> dict[str, Any]:
    """Portable ASR artifact, compatible with --transcripts in main/agent.py."""
    return {
        "id": item["id"],
        "audio": item["audio"],
        "audio_text": item["audio_text"],
        "history_user_asr": [
            {"audio": turn["audio"], "audio_text": turn["audio_text"]}
            for turn in item.get("history", [])
            if turn.get("role") == "user" and turn.get("audio") and turn.get("audio_text")
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-ID streaming LifeMate development pipeline")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--split", choices=("train", "public_test"), default="public_test")
    parser.add_argument("--id", help="run one specific sample")
    parser.add_argument("--limit", type=int, help="maximum number of selected samples")
    parser.add_argument("--asr-model", default="vinai/PhoWhisper-large")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="development OpenRouter model")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--asr-output", type=Path, help="write PhoWhisper transcripts alongside submission output")
    parser.add_argument("--resume", action="store_true", help="append and skip IDs already present in --output")
    args = parser.parse_args()

    load_project_env()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is missing. Put it in the gitignored .env file.")

    split_dir = args.root / args.split
    records = read_jsonl(split_dir / f"{args.split}.jsonl")
    selected = [item for item in records if args.id is None or item["id"] == args.id]
    if args.id and not selected:
        raise SystemExit(f"unknown {args.split} id: {args.id}")
    if args.limit is not None:
        selected = selected[:args.limit]
    completed: set[str] = set()
    if args.resume and args.output.is_file():
        completed = {row["id"] for row in read_jsonl(args.output) if "id" in row}
        selected = [item for item in selected if item["id"] not in completed]
    catalog = catalog_by_name(split_dir / f"{args.split}_tools_catalog.json")

    # Expensive initialization happens once, before the per-ID timer begins.
    transcriber = PhoWhisperTranscriber(args.asr_model, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.asr_output:
        args.asr_output.parent.mkdir(parents=True, exist_ok=True)

    mode = "a" if args.resume else "w"
    asr_stream = args.asr_output.open(mode, encoding="utf-8") if args.asr_output else None
    try:
        with args.output.open(mode, encoding="utf-8") as output:
            for index, item in enumerate(selected, start=1):
                started = time.monotonic()
                enriched = add_transcripts(item, split_dir, transcriber)
                calls, state_hash = infer(
                    enriched,
                    enriched["audio_text"],
                    args.root,
                    args.split,
                    catalog,
                    api_key,
                    args.model,
                )
                row = {"id": enriched["id"], "tool_calls": calls, "state_hash": state_hash}
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                output.flush()  # preserve every completed ID even if a later one fails
                if asr_stream:
                    asr_stream.write(json.dumps(asr_row(enriched), ensure_ascii=False) + "\n")
                    asr_stream.flush()
                elapsed = time.monotonic() - started
                print(f"[{index}/{len(selected)}] {enriched['id']} completed in {elapsed:.2f}s", flush=True)
                print(json.dumps(row, ensure_ascii=False), flush=True)
    finally:
        if asr_stream:
            asr_stream.close()


if __name__ == "__main__":
    main()
