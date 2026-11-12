"""Development agent: transcript -> OpenRouter LLM -> zaic_tools calls.

This script is for prompt and behavior development only. It calls OpenRouter,
which is prohibited in the final offline Docker submission. It never reads a
file under ``<split>_environment`` and uses only ToolEnv's allowed API.

Examples:
    # Test one labelled training item (uses train audio_text).
    .venv/bin/python main/agent.py --split train --id train_0002

    # Produce a development submission from precomputed PhoWhisper transcripts.
    .venv/bin/python main/agent.py --split public_test \
        --transcripts artifacts/public_test_asr_small.jsonl \
        --output artifacts/openrouter_submission.jsonl
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
from typing import Any

from openrouter_test import api_request, load_project_env
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError
from typing import Annotated, Literal


DEFAULT_MODEL = "google/gemma-3-4b-it"
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
    turns: list[dict[str, str]] = []
    for turn in item.get("history", []):
        role = turn.get("role")
        if role == "user" and turn.get("audio_text"):
            turns.append({"role": "user", "text": turn["audio_text"]})
        elif role == "assistant" and turn.get("text"):
            turns.append({"role": "assistant", "text": turn["text"]})
    return turns[-8:]


def decision_prompt(
    item: dict[str, Any], request_text: str, tools: list[dict[str, Any]], trace: list[dict[str, Any]]
) -> str:
    context = dict(item["context"])
    context.pop("installed_apps", None)  # represented precisely by `tools`
    return """Bạn là bộ lập kế hoạch tool cho LifeMate. Hãy thực hiện yêu cầu tiếng Việt bằng cách chọn TỐI ĐA MỘT tool call ở mỗi lượt.

Chỉ được gọi tool trong TOOL_SCHEMAS. Tuân thủ chính xác JSON Schema. Khi cần ID thiết bị, contact, hóa đơn, đơn hàng hoặc dữ liệu app, hãy gọi tool read phù hợp trước; không được đoán ID. Dùng context để suy luận ngày giờ, nhà/công ty và vị trí ("mai" là ngày dương lịch ngay sau current_time). Chỉ truyền tham số optional khi người dùng yêu cầu rõ hoặc nó cần để phân biệt đối tượng; hãy để tool dùng default nếu có. Nếu thiếu tham số bắt buộc không thể suy ra hoặc tra cứu được, gọi sys.ask_missing_info với tool_name và params. Nếu yêu cầu không có app hỗ trợ, hoặc các thao tác đã hoàn tất, chọn done.

Hành động bảo thủ: chỉ gọi write tool khi người dùng yêu cầu rõ ràng. Không tự tạo alarm/reminder/event, đặt xe/đơn hàng, gửi tin, chuyển tiền hay thay đổi app để "giúp" một yêu cầu chỉ đọc thông tin. Khi người dùng nói tự đính chính như "à mà thôi", "không, chỉ..." hoặc "cứ để", phần cuối cùng có hiệu lực và các hành động trước đó bị hủy.

Chỉ trả về JSON hợp lệ, không markdown và không giải thích, đúng một trong hai dạng:
{"action":"call","name":"app.tool","arguments":{}}
{"action":"done"}

USER_CONTEXT:
""" + json.dumps(context, ensure_ascii=False) + "\nCONVERSATION_HISTORY:\n" + json.dumps(
        conversation_summary(item), ensure_ascii=False
    ) + "\nLATEST_REQUEST:\n" + request_text + "\nTOOL_SCHEMAS:\n" + json.dumps(
        tools, ensure_ascii=False, separators=(",", ":")
    ) + "\nEXECUTION_TRACE:\n" + json.dumps(trace, ensure_ascii=False, separators=(",", ":"))


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


def llm_decision(api_key: str, model: str, prompt: str) -> dict[str, Any]:
    response = api_request(api_key, "/chat/completions", {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 512,
        # OpenRouter forwards this OpenAI-compatible structured-output schema
        # to providers that support it. Pydantic validates it again locally.
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "lifemate_agent_decision",
                "strict": True,
                "schema": DECISION_ADAPTER.json_schema(),
            },
        },
    })
    content = response.get("choices", [{}])[0].get("message", {}).get("content")
    if not isinstance(content, str):
        raise ValueError(f"missing text content in LLM response: {response!r}")
    return parse_json_reply(content)


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
    item: dict[str, Any], request_text: str, root: Path, split: str, catalog: dict[str, dict[str, Any]], api_key: str, model: str
) -> tuple[list[dict[str, Any]], str]:
    ToolEnv = import_tool_env(root, split)
    env = ToolEnv(split, item["id"], root=root)  # exactly one environment per sample
    tools = allowed_tool_specs(item, catalog)
    trace: list[dict[str, Any]] = []

    tool_steps = 0
    for _ in range(MAX_LLM_ROUNDS):
        try:
            decision = llm_decision(api_key, model, decision_prompt(item, request_text, tools, trace))
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
    parser = argparse.ArgumentParser(description="Development-only LifeMate OpenRouter agent")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--split", choices=("train", "public_test"), default="train")
    parser.add_argument("--id", help="run one sample and print its prediction")
    parser.add_argument("--transcripts", type=Path, help="ASR JSONL; required for public_test")
    parser.add_argument("--output", type=Path, help="write prediction JSONL when running a split")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    load_project_env()
    import os
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is missing. Put it in the gitignored .env file.")
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

    output_stream = args.output.open("w", encoding="utf-8") if args.output else None
    try:
        for item in selected:
            text = item.get("audio_text")
            if not text:
                raise RuntimeError(f"no transcript for {item['id']}; pass --transcripts generated by main/asr.py")
            calls, state_hash = infer(item, text, args.root, args.split, catalog, api_key, args.model)
            row = {"id": item["id"], "tool_calls": calls, "state_hash": state_hash}
            if output_stream:
                output_stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            else:
                print(json.dumps(row, ensure_ascii=False, indent=2))
    finally:
        if output_stream:
            output_stream.close()


if __name__ == "__main__":
    main()
