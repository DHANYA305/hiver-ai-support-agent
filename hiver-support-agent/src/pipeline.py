"""
End-to-end support agent: one function that takes a raw customer message
and returns intent, a grounded draft reply, and an escalation decision.
"""
from src.escalation.policy import decide_escalation
from src.generation.reply_generator import generate_reply
from src.intent.classifier import load_classifier, predict_intent
from src.retrieval.index import RetrievalIndex
from src.text_utils import clean_tweet


class SupportAgent:
    def __init__(self, classifier=None, index: RetrievalIndex = None):
        self.classifier = classifier or load_classifier()
        self.index = index or RetrievalIndex.load()

    def handle(self, raw_message: str) -> dict:
        text = clean_tweet(raw_message)
        intent = predict_intent(self.classifier, text)
        gen = generate_reply(text, self.index)
        decision = decide_escalation(text, intent, gen["top_similarity"], gen["method"])

        return {
            "input": raw_message,
            "cleaned_text": text,
            "intent": intent,
            "draft_reply": gen["reply"],
            "generation_method": gen["method"],
            "top_similarity": gen["top_similarity"],
            "grounding_examples": gen["grounding_examples"],
            "action": decision["action"],
            "escalate": decision["escalate"],
            "escalation_reason": decision["reason"],
        }


if __name__ == "__main__":
    import json
    agent = SupportAgent()
    examples = [
        "why does my premium account still play ads??",
        "I want a lawyer, you charged me twice and it's fraud",
        "loving the new UI update, great job team!",
        "can you add a sleep timer feature please",
    ]
    for msg in examples:
        out = agent.handle(msg)
        print(json.dumps({k: out[k] for k in ["input", "intent", "action", "escalation_reason", "draft_reply"]}, indent=2))
