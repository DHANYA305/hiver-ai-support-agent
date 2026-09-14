
"""
Draft a reply grounded in how the brand has historically resolved similar
issues.

Two modes (decision_log.md #7):
- offline (default, no API key): take the single most-similar historical
  agent reply as-is if similarity is high, or the best of top-3 with a
  lightweight wrapper otherwise. Fully deterministic.
- llm (GEMINI_API_KEY set): pass the customer message + top-3 retrieved
  questions, resolution pairs to Gemini and ask it to draft a NEW reply
  that follows the brand's tone and resolution pattern, without inventing
  facts not present in the retrieved cases or message itself.

"""
from src.generation.llm_client import complete, llm_available
from src.retrieval.index import RetrievalIndex

_LLM_PROMPT_TMPL = """You are drafting a customer-support reply for the Spotify support team (@SpotifyCares), replying to a tweet.

Customer's message:
"{query}"

Here are the {n} most similar past cases and how support actually resolved them:
{examples}

Write ONE short reply (max 2 sentences, Twitter-support style: friendly, first name if given, ends with a next step like "DM us" or a help-center pointer) that follows the SAME resolution pattern as the examples above. Do not invent account details, refund amounts, or facts not present in the examples or the customer's message. If the examples don't clearly cover this case, give a generic but honest "we'll look into it / DM us" reply instead of guessing.

Reply only with the drafted reply text, nothing else."""


def _format_examples(hits: list[dict]) -> str:
    lines = []
    for i, h in enumerate(hits, 1):
        lines.append(f"{i}. Past customer msg: \"{h['customer_text']}\"\n   Support resolved with: \"{h['agent_text']}\"")
    return "\n".join(lines)


def generate_reply(
    query: str,
    index: RetrievalIndex,
    k: int = 3,
    use_llm: bool = True
) -> dict:
    hits = index.search(query, k=k)
    top_sim = hits[0]["similarity"] if hits else 0.0

    llm_used = False
    if use_llm and llm_available():
        prompt = _LLM_PROMPT_TMPL.format(query=query, n=len(hits), examples=_format_examples(hits))
        llm_reply = complete(prompt, max_tokens=150)
        if llm_reply:
            return {
                "reply": llm_reply,
                "grounding_examples": hits,
                "top_similarity": top_sim,
                "method": "llm_grounded",
            }

    # Offline fallback: extractive, deterministic.
    if not hits:
        reply = "Hey there! Thanks for reaching out — could you DM us more details so we can take a closer look?"
        method = "offline_generic"
    elif top_sim >= 0.22:  # same calibrated threshold as escalation policy's LOW_CONFIDENCE_THRESHOLD
        # Confident match: reuse the historical resolution near-verbatim,
        # since it's a real, brand-approved response to a near-identical issue.
        reply = hits[0]["agent_text"]
        method = "offline_extractive_top1"
    else:
        # Weak match: don't parrot a possibly-irrelevant canned reply —
        # be honest that we're not sure, and point at DM/help-center like
        # the retrieved examples do, without asserting specifics.
        reply = ("Hey! Thanks for flagging this — we want to get it right, so could you "
                 "DM us the details (device, app version, and what you're seeing)? We'll take a look.")
        method = "offline_low_confidence_template"

    return {
        "reply": reply,
        "grounding_examples": hits,
        "top_similarity": top_sim,
        "method": method,
    }


if __name__ == "__main__":
    idx = RetrievalIndex.load()
    out = generate_reply("my downloaded songs disappeared after the update", idx)
    print(out["method"], round(out["top_similarity"], 3))
    print(out["reply"])
