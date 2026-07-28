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
        """Cheap heuristic — no LLM call unless it's plausibly a follow-up.

        Short questions with pronouns or comparatives and no new subject are
        the common case ("what about for men?", "and the side effects?").
        """
        if not self.exchanges:
            return False
        q = question.lower().strip()
        if len(q.split()) > 12:
            return False
        cues = ("what about", "and ", "how about", "why", "them", "it ", "that",
                "those", "he ", "she ", "they", "instead", "also", "more")
        return q.startswith(("and", "what about", "how about", "why", "ok")) or \
            any(c in q for c in cues)


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
