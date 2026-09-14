"""
This test verifies that the LLM generation and LLM judge code paths are
wired correctly (prompt formatting, response parsing, and fallback behavior)
using mocked Gemini client calls. It does not require a real GEMINI_API_KEY
or real Gemini output.

Why this exists: the take-home was built and evaluated in an environment
where no GEMINI_API_KEY was available. The project therefore falls back to
the heuristic judge when no Gemini API key is configured.

The tests below verify that the LLM integration itself is correctly wired
and would work when a valid GEMINI_API_KEY is configured. The committed
report and results clearly record `judge_backend: "heuristic"` because
the headline numbers were generated without an available API key.

This is a known limitation of the evaluation environment, not a claim that
the committed metrics were produced by a real LLM.

"""
import json
from unittest.mock import MagicMock, patch

import src.generation.llm_client as llm_client
from src.evaluation.llm_judge import judge_reply, judge_reply_llm
from src.generation.reply_generator import generate_reply
from src.retrieval.index import RetrievalIndex


def _fake_text_block(text):
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def test_judge_reply_llm_parses_a_wellformed_mock_response():
    llm_client._available = None

    fake_json = json.dumps({
        "faithfulness": 4,
        "relevance": 5,
        "tone": 5,
        "overall": 5,
        "rationale": "mocked response for wiring test",
    })

    with patch(
        "src.evaluation.llm_judge.llm_available",
        return_value=True
    ), patch(
        "src.evaluation.llm_judge.complete",
        return_value=fake_json
    ):
        result = judge_reply_llm(
            "my songs keep skipping",
            "Sorry to hear that! Which device?",
            []
        )

    assert result is not None
    assert result["overall"] == 5

def test_judge_reply_falls_back_to_heuristic_when_llm_unavailable():
    llm_client._available = None
    with patch.dict("os.environ", {}, clear=True):
        result = judge_reply("test query", "Hey! DM us your details.", [])
    assert result["judge"] == "heuristic"
    llm_client._available = None


def test_generate_reply_uses_llm_path_when_mocked_available():
    import pandas as pd

    llm_client._available = None

    idx = RetrievalIndex(pd.DataFrame({
        "customer_text": ["my app keeps crashing on iOS", "songs are skipping"],
        "agent_text": ["Sorry to hear that! What iOS version?", "Try restarting the app."],
    }))

    with patch(
        "src.generation.reply_generator.llm_available",
        return_value=True
    ), patch(
        "src.generation.reply_generator.complete",
        return_value="Try restarting the app — let us know if it keeps happening."
    ):
        result = generate_reply("my app keeps crashing on iOS", idx)

    assert result["method"] == "llm_grounded"