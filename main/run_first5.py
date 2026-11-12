"""Chạy thử OpenRouter trên N mẫu đầu tiên (mặc định 5), không chạy cả split.

Ví dụ (chạy từ thư mục gốc repo):
    export OPENROUTER_API_KEY='...'
    # train có sẵn audio_text và label nên so sánh được tên tool:
    python main/run_first5.py --split train
    # public_test cần transcript ASR (tạo bằng main/asr.py):
    python main/run_first5.py --split public_test --transcripts artifacts/public_test_asr.jsonl
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

from agent import DEFAULT_MODEL, attach_asr_transcripts, catalog_by_name, import_tool_env, infer, read_jsonl
from openrouter_test import load_project_env


def expected_state_hash(item: dict, root: Path, split: str) -> str:
    """Replay the labelled tool calls in a fresh env to get the correct final state hash."""
    ToolEnv = import_tool_env(root, split)
    env = ToolEnv(split, item["id"], root=root)
    for call in item["label"]["tool_calls"]:
        env.call(call["name"], call["arguments"])
    return env.get_state_hash()


def main() -> None:
    parser = argparse.ArgumentParser(description="Test OpenRouter agent on the first N samples")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--split", choices=("train", "public_test"), default="train")
    parser.add_argument("--n", type=int, default=5, help="số mẫu đầu tiên (mặc định 5)")
    parser.add_argument("--transcripts", type=Path, help="ASR JSONL, bắt buộc với public_test")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, help="ghi kết quả ra JSONL (tuỳ chọn)")
    args = parser.parse_args()

    load_project_env()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is missing. Set it in the environment or the gitignored .env file.")

    split_dir = args.root / args.split
    records = read_jsonl(split_dir / f"{args.split}.jsonl")[: args.n]
    if args.transcripts:
        attach_asr_transcripts(records, read_jsonl(args.transcripts))
    catalog = catalog_by_name(split_dir / f"{args.split}_tools_catalog.json")

    rows = []
    name_ok = state_ok = labelled = 0
    for i, item in enumerate(records, start=1):
        item = copy.deepcopy(item)
        text = item.get("audio_text")
        if not text:
            raise SystemExit(f"no transcript for {item['id']}; pass --transcripts (public_test) ")
        calls, state_hash = infer(item, text, args.root, args.split, catalog, api_key, args.model)
        rows.append({"id": item["id"], "tool_calls": calls, "state_hash": state_hash})

        predicted = [c["name"] for c in calls]
        print(f"[{i}/{len(records)}] {item['id']}: {text}")
        print(f"    predicted: {predicted}")
        label = item.get("label")
        if label is not None:
            expected = [c["name"] for c in label["tool_calls"]]
            same_state = state_hash == expected_state_hash(item, args.root, args.split)
            labelled += 1
            name_ok += predicted == expected
            state_ok += same_state
            print(f"    expected : {expected}  tools={'OK' if predicted == expected else 'DIFF'}  state={'OK' if same_state else 'DIFF'}")

    if labelled:
        print(f"\nTổng kết: đúng tên tool {name_ok}/{labelled}, đúng state cuối {state_ok}/{labelled} ({100*state_ok/labelled:.0f}%)")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
