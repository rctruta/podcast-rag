"""Transcript sources — free-first.

Two paths, same output shape, so the rest of the pipeline does not care which
was used:

  youtube  — free, no key, segment-level timestamps (~2-6s granularity).
             Verified 2026-07-27 against a a wellness podcast episode: 1,646 segments,
             10,645 words, 1h03m, zero cost.
  whisper  — paid (~$0.006/min) or local; word-level timestamps. Needed only
             for shows with no captioned video.

Segment-level granularity is sufficient for citation: the goal is to send a
listener to a moment, not to a syllable.

Neither path redistributes transcripts. This fetches at index time on the
user's machine, for a corpus the user chooses.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Segment:
    """Normalised transcript unit. Both sources produce these."""
    text: str
    start: float
    duration: float

    @property
    def end(self) -> float:
        return self.start + self.duration


def from_youtube(video_id: str, languages: tuple[str, ...] = ("en",)) -> list[Segment]:
    """Free path. Raises if the video has no captions in `languages`."""
    from youtube_transcript_api import YouTubeTranscriptApi

    fetched = YouTubeTranscriptApi().fetch(video_id, languages=list(languages))
    return [Segment(text=s.text.replace("\n", " ").strip(),
                    start=float(s.start), duration=float(s.duration))
            for s in fetched.snippets if s.text.strip()]


def list_available(video_id: str) -> list[dict]:
    """What captions exist, before committing to a fetch."""
    from youtube_transcript_api import YouTubeTranscriptApi

    return [{"language_code": t.language_code, "generated": t.is_generated}
            for t in YouTubeTranscriptApi().list(video_id)]


def segments_to_words(segments: list[Segment]) -> list[dict]:
    """Adapt segments to the word-shape `chunk_words` expects.

    Timestamps are interpolated evenly across a segment's words. That is an
    approximation — a word's true offset may be off by a second or two inside
    its segment — and it is honest to say so: it is accurate to the SEGMENT,
    which is the unit the source actually provides. Chunk boundaries still land
    on real segment boundaries because chunking respects word order.
    """
    words: list[dict] = []
    for seg in segments:
        toks = seg.text.split()
        if not toks:
            continue
        step = seg.duration / len(toks) if seg.duration > 0 else 0.0
        for i, tok in enumerate(toks):
            words.append({"word": tok,
                          "start": seg.start + i * step,
                          "end": seg.start + (i + 1) * step})
    return words
