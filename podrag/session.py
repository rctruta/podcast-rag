"""Conversation memory.

Adapted in concept from the DeepLearning.AI RAG chatbot's SessionManager: keep
a bounded exchange history so follow-ups resolve ("what about for men?").

Deliberately NOT adopted from that codebase: tool-calling retrieval, where the
model decides whether to search. That would convert the refusal gate from a
mechanical check into an advisory one, which is the property this project is
built around.

History is used to REWRITE the follow-up into a standalone question before
retrieval, rather than being stuffed into the synthesis prompt. Retrieval needs
a self-contained query; "what about for men?" embeds to nothing useful.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Exchange:
    question: str
    answer: str
    refused: bool = False


@dataclass
class Session:
    max_history: int = 5
    exchanges: list[Exchange] = field(default_factory=list)

    def add(self, question: str, answer: str, refused: bool = False) -> None:
        self.exchanges.append(Exchange(question, answer, refused))
        if len(self.exchanges) > self.max_history:
            self.exchanges = self.exchanges[-self.max_history:]

    def clear(self) -> None:
        self.exchanges.clear()

    def as_context(self) -> str:
        if not self.exchanges:
            return ""
        return "\n".join(
            f"Q: {e.question}\nA: {(e.answer or '(refused)')[:300]}"
            for e in self.exchanges)

    def looks_like_followup(self, question: str) -> bool:
        """Conservative: only LEADING cues count, and only for short questions.

        An earlier version substring-matched cues like "why", "that", "more",
        "it " anywhere in the question, which swept up plenty of standalone
        questions ("why do mitochondria matter") and silently glued them to
        prior context. False positives are worse than false negatives here: a
        missed follow-up still searches literally and returns something, while
        a false positive rewrites a fresh question into the previous topic and
        the user cannot tell why the answer drifted.
        """
        if not self.exchanges:
            return False
        q = question.lower().strip().rstrip("?")
        if len(q.split()) > 8:          # a long question carries its own subject
            return False
        leading = ("and ", "what about", "how about", "but ", "also ",
                   "plus ", "ok ", "okay ", "so ")
        if q.startswith(leading):
            return True
        # bare pronoun subjects with no noun of their own
        bare = ("what about it", "does it", "is it", "are they", "do they",
                "does he", "does she", "and them", "what else",
                "why is that", "why is it", "why does that", "why not",
                "why though", "how so")
        if any(q.startswith(b) for b in bare):
            return True
        # "why?" / "why is that" style: leading why with no subject of its own
        return q.startswith("why") and len(q.split()) <= 3


def standalone_question(session: Session, question: str, model: str,
                        api_key: str | None) -> tuple[str, bool]:
    """Rewrite a follow-up into a self-contained question. Returns
    (question, was_rewritten). Falls back to the original on any failure —
    a rewrite that fails should degrade to normal search, not break it."""
    if not api_key or not session.looks_like_followup(question):
        return question, False
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        r = client.chat.completions.create(
            model=model, temperature=0, max_tokens=100,
            messages=[
                {"role": "system", "content":
                 "Rewrite the user's latest question as a standalone question "
                 "using the conversation for context. Keep it short. Output "
                 "only the rewritten question, nothing else."},
                {"role": "user", "content":
                 f"Conversation:\n{session.as_context()}\n\nLatest: {question}"},
            ])
        out = (r.choices[0].message.content or "").strip().strip('"')
        return (out, True) if out else (question, False)
    except Exception:
        return question, False
