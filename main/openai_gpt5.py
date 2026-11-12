"""Development-only OpenAI GPT-5 client (Chat Completions API).

Never put an API key in source code. Set ``OPENAI_API_KEY`` in the shell or in
the gitignored repository-root ``.env`` file. Final competition inference must
run offline, so this is for development only.

Example:
    export OPENAI_API_KEY='...'
    .venv/bin/python main/openai_gpt5.py --prompt 'Trả lời đúng một từ: OK'
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-5"
MAX_TOKENS = 10_000  # trần cứng cho mỗi lần gọi (gồm cả token suy luận ẩn)


def load_project_env() -> None:
    """Load OPENAI_API_KEY from the repository-root .env (shell env wins)."""
    path = Path(__file__).resolve().parents[1] / ".env"
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() == "OPENAI_API_KEY":
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def api_request(api_key: str, path: str, payload: dict[str, Any] | None = None, timeout: int = 120) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = Request(
        f"{BASE_URL}{path}",
        data=data,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST" if data is not None else "GET",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")[:2_000]
        raise RuntimeError(f"OpenAI returned HTTP {error.code}: {detail}") from error
    except URLError as error:
        raise RuntimeError(f"Cannot connect to OpenAI: {error.reason}") from error


def chat(
    api_key: str,
    model: str,
    prompt: str,
    *,
    max_completion_tokens: int = MAX_TOKENS,
    reasoning_effort: str = "minimal",
    response_format: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One GPT-5 chat completion.

    GPT-5 differs from older chat models: it rejects ``temperature`` /
    ``max_tokens`` (use ``max_completion_tokens``) and spends part of that
    budget on hidden reasoning tokens, so keep the budget generous and the
    reasoning effort low for simple tool-planning calls.
    """
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_completion_tokens": min(max_completion_tokens, MAX_TOKENS),
        "reasoning_effort": reasoning_effort,
    }
    if response_format is not None:
        payload["response_format"] = response_format
    return api_request(api_key, "/chat/completions", payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Development-only GPT-5 test")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default="Trả lời đúng một từ: OK")
    args = parser.parse_args()

    load_project_env()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Set OPENAI_API_KEY in the environment or .env; do not save it in a source file.")
    response = chat(api_key, args.model, args.prompt, max_completion_tokens=256)
    print(json.dumps({
        "model": response.get("model", args.model),
        "text": response.get("choices", [{}])[0].get("message", {}).get("content"),
        "usage": response.get("usage"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
