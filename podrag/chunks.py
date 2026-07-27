"""Timestamp-preserving chunking.

THE INVARIANT THIS FILE EXISTS TO ENFORCE: a chunk that cannot state where it
came from is not a chunk. Every transform carries (episode guid, start, end)
through, so a retrieved passage can always be turned into a playable citation.

Chunking on word timestamps rather than characters is what makes that possible —
character offsets cannot be mapped back to audio.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class Chunk:
    show: str
    episode_guid: str
    episode_title: str
    published: str
    chunk_index: int
    text: str
    start_s: float
    end_s: float
    speaker: str | None = None

    @property
    def timestamp(self) -> str:
        m, s = divmod(int(self.start_s), 60)
        h, m = divmod(m, 60)
        return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"

    def citation(self) -> str:
        return f"{self.episode_title} · {self.timestamp}"

    def as_dict(self) -> dict:
        d = asdict(self)
        d["timestamp"] = self.timestamp
        return d


def chunk_words(words: list[dict], *, show: str, episode_guid: str,
                episode_title: str, published: str,
                target_words: int = 180, overlap_words: int = 30) -> list[Chunk]:
    """Group Whisper word-timestamps into overlapping chunks.

    `words` is [{"word": str, "start": float, "end": float}, ...] — the shape
    Whisper emits with word_timestamps=True.

    Overlap exists so a passage split across a boundary is still retrievable
    whole from at least one chunk.
    """
    if not words:
        return []
    if overlap_words >= target_words:
        raise ValueError("overlap_words must be smaller than target_words")

    chunks: list[Chunk] = []
    step = target_words - overlap_words
    idx = 0
    for start_i in range(0, len(words), step):
        window = words[start_i:start_i + target_words]
        if not window:
            break
        text = " ".join(w["word"].strip() for w in window).strip()
        if not text:
            continue
        chunks.append(Chunk(
            show=show,
            episode_guid=episode_guid,
            episode_title=episode_title,
            published=published,
            chunk_index=idx,
            text=text,
            start_s=float(window[0]["start"]),
            end_s=float(window[-1]["end"]),
        ))
        idx += 1
        if start_i + target_words >= len(words):
            break
    return chunks


def deep_link(chunk: Chunk) -> str:
    """A citation you can click. Falls back to the audio URL with a time
    fragment when no episode page is known."""
    return f"{chunk.episode_guid}#t={int(chunk.start_s)}"
