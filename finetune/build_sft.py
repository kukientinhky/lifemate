"""Build step-level SFT data from train/train.jsonl.

Each training sample becomes one example per agent step, using the SAME
decision_prompt() the agent uses at inference, so train/test formats match.
Run from the lifemate/ directory:  python finetune/build_sft.py
"""
import json, random, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "main"))
from agent import allowed_tool_specs, catalog_by_name, decision_prompt, read_jsonl  # noqa: E402

rows = read_jsonl(ROOT / "train" / "train.jsonl")
catalog = catalog_by_name(ROOT / "train" / "train_tools_catalog.json")

def examples_for(item):
    tools = allowed_tool_specs(item, catalog)
    calls, results = item["label"]["tool_calls"], item["label"]["tool_results"]
    trace = []
    for call, res in zip(calls, results):
        decision = {"action": "call", "name": call["name"], "arguments": call["arguments"]}
        yield decision_prompt(item, item["audio_text"], tools, trace), decision
        trace.append({"tool_call": decision, "tool_result": res})
    if not calls or calls[-1]["name"] != "sys.ask_missing_info":  # asking ends the turn
        yield decision_prompt(item, item["audio_text"], tools, trace), {"action": "done"}

random.seed(0)
ids = sorted(r["id"] for r in rows)
random.shuffle(ids)
val_ids = set(ids[: len(ids) // 10])

out = ROOT / "finetune" / "out"; out.mkdir(exist_ok=True)
files = {"train": open(out / "sft_train.jsonl", "w", encoding="utf-8"),
         "val": open(out / "sft_val.jsonl", "w", encoding="utf-8")}
n = {"train": 0, "val": 0}; maxlen = 0
for item in rows:
    split = "val" if item["id"] in val_ids else "train"
    for prompt, decision in examples_for(item):
        maxlen = max(maxlen, len(prompt))
        files[split].write(json.dumps({"id": item["id"], "prompt": prompt,
            "completion": json.dumps(decision, ensure_ascii=False, separators=(",", ":"))}, ensure_ascii=False) + "\n")
        n[split] += 1
(out / "val_ids.json").write_text(json.dumps(sorted(val_ids)))
print(n, "max prompt chars:", maxlen)
