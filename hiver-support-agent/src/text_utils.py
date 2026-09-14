import html
import re

URL_RE = re.compile(r"https?://\S+")
MENTION_RE = re.compile(r"@\w+")
WHITESPACE_RE = re.compile(r"\s+")


def clean_tweet(text: str, strip_mentions: bool = True) -> str:
    """Normalize a raw tweet: unescape HTML entities, drop URLs, optionally
    drop @mentions (Twitter support threads open with @handle), collapse
    whitespace. Deliberately conservative — we don't lowercase or strip
    punctuation here because the LLM-facing steps want natural text; the
    TF-IDF steps do their own lowercasing/tokenizing downstream."""
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = URL_RE.sub("", text)
    if strip_mentions:
        text = MENTION_RE.sub("", text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


def is_low_signal(text: str, min_chars: int = 8) -> bool:
    """Filter out near-empty messages (e.g. a message that was only a
    @mention + emoji) that can't be usefully classified or retrieved on."""
    stripped = re.sub(r"[^a-zA-Z0-9]", "", text)
    return len(stripped) < min_chars
