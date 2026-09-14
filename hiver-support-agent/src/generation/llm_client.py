"""
Thin optional Gemini LLM wrapper.

If GEMINI_API_KEY is configured, generation and evaluation can use Gemini.
Without an API key, callers fall back to their deterministic methods.
"""

import os
import time
from dotenv import load_dotenv

load_dotenv()

_client = None
_available = None
_last_request_time = 0
MIN_REQUEST_INTERVAL = 15


def llm_available() -> bool:
    global _available, _client

    if _available is not None:
        return _available

    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        _available = False
        return False

    try:
        from google import genai

        _client = genai.Client(api_key=api_key)
        _available = True

    except Exception as e:
        print(f"[llm_client] Gemini setup failed: {e}")
        _available = False

    return _available


def complete(prompt: str, max_tokens: int = 400) -> str | None:
    if not llm_available():
        return None

    try:
        resp = _client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=prompt,
        )

        return resp.text.strip() if resp and resp.text else None

    except Exception as err:
        print(f"[llm_client] Gemini call failed: {err}")
        return None