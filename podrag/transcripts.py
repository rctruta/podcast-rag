"""Transcript segments — the interface every source produces.

`Segment` is what sources agree on, so the rest of the pipeline does not care
where a transcript came from. Sources:

  rss_transcripts.py — publisher-provided <podcast:transcript> (the default)
  whisper            — for shows publishing audio but no transcript

A YouTube caption path was REMOVED 2026-07-28. Verified, not assumed:
the official `captions.download` "requires the user to have permission to edit
the video", so there is no sanctioned way to fetch a third party's captions
and a third-party caption library necessarily uses an undocumented endpoint. YouTube's
ToS prohibits automated scraping and circumvention, and the prohibition is on
ACCESS — keeping the data local does not cure it. Every episode previously
indexed reported license: "youtube", so no Creative Commons exemption applied.

That removing it required no downstream change is the point of this interface.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Segment:
    """Normalised transcript unit. Every source produces these."""
    text: str
    start: float
    duration: float

    @property
    def end(self) -> float:
        return self.start + self.duration


def segments_to_words(segments: list[Segment]) -> list[dict]:
    """Adapt segments to the word-shape `chunk_words` expects.

    Timestamps are interpolated evenly across a segment's words. That is an
    approximation and it is honest to say so: it is accurate to the SEGMENT,
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
