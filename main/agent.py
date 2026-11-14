"""Agent: transcript -> local Qwen3.5-4B -> zaic_tools calls.

The LLM runs locally (no API), so it is usable offline. It never reads a
file under ``<split>_environment`` and uses only ToolEnv's allowed API.

Examples:
    # Test one labelled training item (uses train audio_text).
    .venv/bin/python main/agent.py --split train --id train_0002

    # Produce a development submission from precomputed PhoWhisper transcripts.
    .venv/bin/python main/agent.py --split public_test \
        --transcripts artifacts/public_test_asr_small.jsonl \
        --output artifacts/qwen35_submission.jsonl
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
from typing import Any

import qwen_local
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError
from typing import Annotated, Literal


DEFAULT_MODEL = qwen_local.MODEL_ID  # Qwen/Qwen3.5-4B, local weights
MAX_TOOL_STEPS = 8
MAX_LLM_ROUNDS = 12


class ToolCallDecision(BaseModel):
    """The only JSON shape that is allowed to request a tool execution."""

    model_config = ConfigDict(extra="forbid")

    action: Literal["call"]
    name: str = Field(min_length=1)
    arguments: dict[str, Any]


class DoneDecision(BaseModel):
    """The only JSON shape that is allowed to finish an agent turn."""

    model_config = ConfigDict(extra="forbid")

    action: Literal["done"]


Decision = Annotated[ToolCallDecision | DoneDecision, Field(discriminator="action")]
DECISION_ADAPTER = TypeAdapter(Decision)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def catalog_by_name(path: Path) -> dict[str, dict[str, Any]]:
    catalog = json.loads(path.read_text(encoding="utf-8"))
    return {tool["name"]: tool for app in catalog["apps"] for tool in app.get("tools", [])}


def allowed_tool_specs(item: dict[str, Any], catalog: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    allowed = {
        tool_name
        for app in item["context"]["installed_apps"].values()
        for tool_name in app.get("tools", [])
    }
    allowed.add("sys.ask_missing_info")
    # Keep exactly the information the LLM needs, rather than the entire
    # catalog. This keeps the final local-agent prompt within its 16K budget.
    return [
        {
            "name": catalog[name]["name"],
            "kind": catalog[name]["kind"],
            "description": catalog[name].get("description", ""),
            "parameters": catalog[name].get("parameters", {}),
        }
        for name in sorted(allowed)
        if name in catalog
    ]


def conversation_summary(item: dict[str, Any]) -> list[dict[str, str]]:
    """Use only prior conversation supplied in the JSONL, never app state."""
    turns: list[dict[str, Any]] = []
    for turn in item.get("history", []):
        role = turn.get("role")
        if role == "user" and turn.get("audio_text"):
            turns.append({"role": "user", "text": turn["audio_text"]})
        elif role == "assistant" and turn.get("text"):
            entry: dict[str, Any] = {"role": "assistant", "text": turn["text"]}
            # Keep the tools the assistant already ran (ids, dates) so that
            # references like "buổi đấy" / "cái vừa rồi" can be resolved.
            calls, results = turn.get("tool_calls") or [], turn.get("tool_results") or []
            if calls:
                entry["tools"] = [
                    {
                        "call": {"name": call.get("name"), "arguments": call.get("arguments", {})},
                        "result": json.dumps(results[i], ensure_ascii=False, separators=(",", ":"))[:600]
                        if i < len(results)
                        else None,
                    }
                    for i, call in enumerate(calls)
                ]
            turns.append(entry)
    return turns[-8:]


def time_hints(context: dict[str, Any]) -> dict[str, Any]:
    """Pre-compute calendar facts so the model never has to do date arithmetic."""
    from datetime import datetime, timedelta

    try:
        now = datetime.fromisoformat(context["current_time"])
    except (KeyError, ValueError):
        return {}
    names = ["Thứ hai", "Thứ ba", "Thứ tư", "Thứ năm", "Thứ sáu", "Thứ bảy", "Chủ nhật"]
    days = {}
    for offset in range(10):
        day = now + timedelta(days=offset)
        label = names[day.weekday()] + {0: " (hôm nay)", 1: " (ngày mai)", 2: " (ngày kia)"}.get(offset, "")
        days[day.strftime("%Y-%m-%d")] = label
    return {"gio_hien_tai": now.strftime("%H:%M"), "lich_10_ngay_toi": days}


PROMPT_RULES_PATH = Path(__file__).with_name("prompt_rules.txt")


def load_prompt_rules() -> str:
    """Rules/format text that opens every prompt. Edit main/prompt_rules.txt (not this file) to tune it.

    Training data (finetune/build_sft.py) is built from this same text, so after
    editing it you must rebuild the SFT data and re-train for the change to match.
    """
    return PROMPT_RULES_PATH.read_text(encoding="utf-8")


def decision_prompt(
    item: dict[str, Any], request_text: str, tools: list[dict[str, Any]], trace: list[dict[str, Any]]
) -> str:
    context = dict(item["context"])
    context.pop("installed_apps", None)  # represented precisely by `tools`
    context["time_hints"] = time_hints(context)
    # Static part first (rules + schemas), per-request part last, so the
    # latest request and trace are the freshest thing the model reads.
    return load_prompt_rules().rstrip("\n") + "\n\nTOOL_SCHEMAS:\n" + json.dumps(tools, ensure_ascii=False, separators=(",", ":")) + "\n\nUSER_CONTEXT:\n" + json.dumps(
        context, ensure_ascii=False
    ) + "\nCONVERSATION_HISTORY:\n" + json.dumps(
        conversation_summary(item), ensure_ascii=False
    ) + "\nLATEST_REQUEST:\n" + request_text + "\nEXECUTION_TRACE:\n" + json.dumps(
        trace, ensure_ascii=False, separators=(",", ":")
    ) + "\nQUYẾT ĐỊNH TIẾP THEO (JSON):"


def parse_json_reply(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"LLM did not return a JSON object: {content[:300]!r}")
    try:
        decision = DECISION_ADAPTER.validate_json(text[start : end + 1])
    except (ValidationError, ValueError) as error:
        raise ValueError(f"invalid agent decision JSON: {error}") from error
    return decision.model_dump()


def llm_decision(model: str, prompt: str) -> dict[str, Any]:
    return parse_json_reply(qwen_local.generate(prompt, model))


def import_tool_env(root: Path, split: str):
    split_path = str(root / split)
    if split_path not in sys.path:
        sys.path.insert(0, split_path)
    from zaic_tools import ToolEnv
    return ToolEnv


def attach_asr_transcripts(records: list[dict[str, Any]], transcript_rows: list[dict[str, Any]]) -> None:
    """Attach current and historical user transcripts to their JSONL records."""
    by_id = {row["id"]: row for row in transcript_rows}
    for item in records:
        row = by_id.get(item["id"])
        if not row:
            continue
        item["audio_text"] = row.get("audio_text", item.get("audio_text"))
        history_text = {entry.get("audio"): entry.get("audio_text") for entry in row.get("history_user_asr", [])}
        for turn in item.get("history", []):
            if turn.get("role") == "user" and turn.get("audio") in history_text:
                turn["audio_text"] = history_text[turn["audio"]]


def infer(
    item: dict[str, Any], request_text: str, root: Path, split: str, catalog: dict[str, dict[str, Any]], model: str
) -> tuple[list[dict[str, Any]], str]:
    ToolEnv = import_tool_env(root, split)
    env = ToolEnv(split, item["id"], root=root)  # exactly one environment per sample
    tools = allowed_tool_specs(item, catalog)
    trace: list[dict[str, Any]] = []

    tool_steps = 0
    for _ in range(MAX_LLM_ROUNDS):
        try:
            decision = llm_decision(model, decision_prompt(item, request_text, tools, trace))
        except (ValueError, json.JSONDecodeError) as error:
            trace.append({"response_format_error": str(error)})
            continue
        if decision["action"] == "done":
            break
        check = env.validate(decision["name"], decision["arguments"])
        if not check["ok"]:
            # validate is non-mutating and unlogged, so the model gets a chance
            # to correct its JSON without submitting a rejected tool call.
            trace.append({"proposed_call": decision, "validation_error": check["error"]})
            continue
        result = env.call(decision["name"], decision["arguments"])
        trace.append({"tool_call": decision, "tool_result": result})
        tool_steps += 1
        if decision["name"] == "sys.ask_missing_info":
            break  # asking the user ends the turn; labels never continue after it
        if tool_steps >= MAX_TOOL_STEPS:
            print(
                f"warning: {item['id']} reached the {MAX_TOOL_STEPS}-tool safety cap; submitting completed calls",
                file=sys.stderr,
            )
            break
    else:
        print(
            f"warning: {item['id']} reached the {MAX_LLM_ROUNDS}-round planning cap; submitting completed calls",
            file=sys.stderr,
        )

    return env.get_tool_calls(), env.get_state_hash()


def main() -> None:
    parser = argparse.ArgumentParser(description="LifeMate agent using local Qwen3.5-4B")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--split", choices=("train", "public_test"), default="train")
    parser.add_argument("--id", help="run one sample and print its prediction")
    parser.add_argument("--transcripts", type=Path, help="ASR JSONL; required for public_test")
    parser.add_argument("--output", type=Path, help="write prediction JSONL when running a split")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--resume", action="store_true", help="append to --output and skip IDs already in it")
    args = parser.parse_args()

    if not args.id and not args.output:
        raise SystemExit("provide --id for one sample or --output for a split")

    split_dir = args.root / args.split
    records = read_jsonl(split_dir / f"{args.split}.jsonl")
    transcript_rows: list[dict[str, Any]] = []
    if args.transcripts:
        transcript_rows = read_jsonl(args.transcripts)
        attach_asr_transcripts(records, transcript_rows)
    catalog = catalog_by_name(split_dir / f"{args.split}_tools_catalog.json")
    selected = [copy.deepcopy(item) for item in records if args.id is None or item["id"] == args.id]
    if args.id and not selected:
        raise SystemExit(f"unknown {args.split} id: {args.id}")

    if args.resume and args.output and args.output.is_file():
        done = {row["id"] for row in read_jsonl(args.output) if "id" in row}
        selected = [item for item in selected if item["id"] not in done]
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
    output_stream = args.output.open("a" if args.resume else "w", encoding="utf-8") if args.output else None
    try:
        for index, item in enumerate(selected, start=1):
            text = item.get("audio_text")
            if not text:
                raise RuntimeError(f"no transcript for {item['id']}; pass --transcripts generated by main/asr.py")
            try:
                calls, state_hash = infer(item, text, args.root, args.split, catalog, args.model)
            except Exception as error:  # one bad sample must not stop a full run
                print(f"[{index}/{len(selected)}] {item['id']} FAILED: {error}", file=sys.stderr, flush=True)
                continue  # no row written, so --resume retries it
            row = {"id": item["id"], "tool_calls": calls, "state_hash": state_hash}
            if output_stream:
                output_stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                output_stream.flush()
                print(f"[{index}/{len(selected)}] {item['id']}: {[c['name'] for c in calls]}", flush=True)
            else:
                print(json.dumps(row, ensure_ascii=False, indent=2))
    finally:
        if output_stream:
            output_stream.close()


if __name__ == "__main__":
    main()
