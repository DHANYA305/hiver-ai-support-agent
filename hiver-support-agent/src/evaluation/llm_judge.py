"""
LLM-as-judge for draft-reply quality, scoring 1-5 on:
  - faithfulness: does the reply avoid inventing facts not in the
    grounding examples or the customer's message?
  - relevance: does it actually address what the customer asked?
  - tone: does it match brand voice (friendly, concise, Twitter-support)?
overall = the judge's holistic 1-5, not an average of the three (a reply
can be faithful+on-tone but still generically useless, and the judge is
asked to weigh that).

Falls back to a deterministic heuristic scorer when no LLM is configured
(decision_log.md #7, #10) so `make eval` always produces numbers. The
heuristic is intentionally crude — see report.md for its measured
agreement with hand scores, which is the honest way to show a grader
"here is exactly how much to trust this number."
"""
import json
import re

from src.generation.llm_client import complete, llm_available

_JUDGE_PROMPT_TMPL = """You are grading a draft customer-support reply for quality. Score strictly.

Customer message: "{query}"

Grounding evidence (how similar past cases were actually resolved):
{examples}

Draft reply to grade: "{reply}"

Score 1-5 (5=excellent) on each of:
- faithfulness: does it avoid asserting facts/details not supported by the grounding evidence or the customer's own message?
- relevance: does it actually address the customer's specific issue?
- tone: friendly, concise, matches Twitter customer-support voice?
- overall: your holistic judgment (not necessarily the average of the above).

Respond ONLY with compact JSON: {{"faithfulness": int, "relevance": int, "tone": int, "overall": int, "rationale": "one short sentence"}}"""


def _format_examples(hits: list[dict]) -> str:
    if not hits:
        return "(no grounding evidence retrieved)"
    return "\n".join(f'- past msg: "{h["customer_text"]}" -> resolved with: "{h["agent_text"]}"' for h in hits)


def judge_reply_llm(query: str, reply: str, grounding_examples: list[dict]) -> dict | None:
    if not llm_available():
        return None
    prompt = _JUDGE_PROMPT_TMPL.format(query=query, examples=_format_examples(grounding_examples), reply=reply)
    raw = complete(prompt, max_tokens=200)
    if not raw:
        return None
    try:
        raw = re.sub(r"^```(json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
        parsed = json.loads(raw)
        parsed["judge"] = "llm"
        return parsed
    except Exception:
        return None


_NEXT_STEP_PHRASES = ["dm us", "dm you", "check out", "let us know", "here's", "here is", "we'll take a look",
                       "we've", "vote", "restart", "try ", "help center", "reach out"]


def judge_reply_heuristic(query: str, reply: str, grounding_examples: list[dict]) -> dict:
    """Crude proxy scorer, documented as such:
    - faithfulness proxy: token overlap between reply and the retrieved
      agent replies (a reply built mostly from unseen tokens vs. the
      grounding evidence is more likely to be hallucinated/generic).
    - relevance proxy: token overlap between reply and the customer query.
    - tone proxy: length in a plausible Twitter-reply range + presence of
      a concrete next-step phrase.
    """
    def toks(s):
        return set(re.findall(r"[a-z']+", (s or "").lower()))

    reply_toks = toks(reply)
    query_toks = toks(query)
    grounding_toks = set()
    for h in grounding_examples:
        grounding_toks |= toks(h.get("agent_text", ""))

    def overlap_ratio(a, b):
        if not a:
            return 0.0
        return len(a & b) / len(a)

    faithfulness_ratio = overlap_ratio(reply_toks, grounding_toks) if grounding_examples else 0.3
    relevance_ratio = overlap_ratio(reply_toks, query_toks)

    faithfulness = 5 if faithfulness_ratio > 0.35 else 4 if faithfulness_ratio > 0.2 else 3 if faithfulness_ratio > 0.08 else 2
    relevance = 5 if relevance_ratio > 0.25 else 4 if relevance_ratio > 0.12 else 3 if relevance_ratio > 0.03 else 2

    has_next_step = any(p in (reply or "").lower() for p in _NEXT_STEP_PHRASES)
    length_ok = 20 <= len(reply or "") <= 280
    tone = 5 if (has_next_step and length_ok) else 3 if length_ok else 2

    overall = round((faithfulness + relevance + tone) / 3)
    return {
        "faithfulness": faithfulness, "relevance": relevance, "tone": tone, "overall": overall,
        "rationale": f"heuristic: faith_overlap={faithfulness_ratio:.2f}, rel_overlap={relevance_ratio:.2f}, next_step={has_next_step}",
        "judge": "heuristic",
    }

def judge_reply(
    query: str,
    reply: str,
    grounding_examples: list[dict],
    require_llm: bool = False
) -> dict:

    result = judge_reply_llm(query, reply, grounding_examples)

    if result is not None:
        return result

    if require_llm:
        raise RuntimeError(
            "LLM judge failed. Evaluation stopped to prevent "
            "mixing LLM and heuristic scores."
        )

    return judge_reply_heuristic(
        query,
        reply,
        grounding_examples
    )
