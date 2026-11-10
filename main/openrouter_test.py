"""Development-only OpenRouter connectivity and chat test.

Never put an API key in source code, a notebook, a command committed to git,
or the final competition Docker image. External APIs are useful during
development only; LifeMate final inference must run offline.

Examples:
    export OPENROUTER_API_KEY='...'
    .venv/bin/python main/openrouter_test.py --list-models --filter qwen
    .venv/bin/python main/openrouter_test.py --model <model-id> \
        --prompt 'Trả lời đúng một từ: OK'
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


BASE_URL = "https://openrouter.ai/api/v1"


def load_project_env() -> None:
    """Load only the local key from the repository-root .env file.

    The file is gitignored and environment variables already supplied by the
    shell take precedence. This intentionally avoids a new runtime dependency.
    """
    path = Path(__file__).resolve().parents[1] / ".env"
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() == "OPENROUTER_API_KEY":
            os.environ.setdefault(key.strip(), value.strip().strip("\"'") )


def api_request(api_key: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = Request(
        f"{BASE_URL}{path}",
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost",
            "X-Title": "LifeMate development test",
        },
        method="POST" if data is not None else "GET",
    )
    try:
        with urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        # Do not print request headers or environment variables: they contain
        # the secret. The API response body itself contains useful safe detail.
        detail = error.read().decode("utf-8", errors="replace")[:2_000]
        raise RuntimeError(f"OpenRouter returned HTTP {error.code}: {detail}") from error
    except URLError as error:
        raise RuntimeError(f"Cannot connect to OpenRouter: {error.reason}") from error


def list_models(api_key: str, name_filter: str) -> None:
    models = api_request(api_key, "/models").get("data", [])
    query = name_filter.casefold()
    rows = [
        {
            "id": model.get("id"),
            "name": model.get("name"),
            "context_length": model.get("context_length"),
        }
        for model in models
        if not query or query in str(model.get("id", "")).casefold() or query in str(model.get("name", "")).casefold()
    ]
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    print(f"{len(rows)} model(s)", file=sys.stderr)


def chat_test(api_key: str, model: str, prompt: str, max_tokens: int) -> None:
    response = api_request(api_key, "/chat/completions", {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
    })
    choice = response.get("choices", [{}])[0]
    content = choice.get("message", {}).get("content")
    print(json.dumps({
        "model": response.get("model", model),
        "text": content,
        "usage": response.get("usage"),
    }, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Development-only OpenRouter test")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--list-models", action="store_true", help="verify the key and list available models")
    action.add_argument("--model", help="model ID for one deterministic chat-completion test")
    parser.add_argument("--filter", default="", help="case-insensitive filter for --list-models")
    parser.add_argument("--prompt", default="Trả lời đúng một từ: OK", help="test prompt for --model")
    parser.add_argument("--max-tokens", type=int, default=32)
    args = parser.parse_args()

    load_project_env()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("Set OPENROUTER_API_KEY in the environment; do not save it in a source file.")
    if args.list_models:
        list_models(api_key, args.filter)
    else:
        chat_test(api_key, args.model, args.prompt, args.max_tokens)


if __name__ == "__main__":
    main()
