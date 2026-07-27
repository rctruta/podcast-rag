"""CLI: index a show, then ask it questions."""
from __future__ import annotations

import argparse
import sys


def main() -> int:
    p = argparse.ArgumentParser(prog="podrag")
    sub = p.add_subparsers(dest="cmd", required=True)

    ix = sub.add_parser("index", help="index YouTube episodes by video id")
    ix.add_argument("video_ids", nargs="+")
    ix.add_argument("--show", required=True)
    ix.add_argument("--title", default=None, help="episode title (single video)")
    ix.add_argument("--db", default="./podrag.lance")
    ix.add_argument("--add", action="store_true", help="append instead of overwrite")

    q = sub.add_parser("ask", help="ask a question")
    q.add_argument("question")
    q.add_argument("--db", default="./podrag.lance")
    q.add_argument("--k", type=int, default=5)
    q.add_argument("--search", default="hybrid", choices=["vector", "keyword", "hybrid"])
    q.add_argument("--model", default="gpt-4o-mini")
    q.add_argument("--min-confidence", type=float, default=None)
    q.add_argument("--no-synth", action="store_true", help="retrieval only, no LLM call")

    a = p.parse_args()

    if a.cmd == "index":
        from podrag.chunks import chunk_words
        from podrag.index import build_index, index_version
        from podrag.transcripts import from_youtube, segments_to_words

        chunks = []
        for vid in a.video_ids:
            try:
                segs = from_youtube(vid)
            except Exception as e:
                print(f"  {vid}: no transcript ({type(e).__name__})", file=sys.stderr)
                continue
            chunks += chunk_words(segments_to_words(segs), show=a.show,
                                  episode_guid=vid,
                                  episode_title=(a.title or vid),
                                  published="")
            print(f"  {vid}: {len(segs)} segments")
        if not chunks:
            print("nothing indexed", file=sys.stderr)
            return 1
        tbl = build_index(chunks, db_path=a.db, mode="add" if a.add else "overwrite")
        print(f"indexed {len(chunks)} chunks · version {index_version(tbl)}")
        return 0

    from podrag.answer import DEFAULT_MIN_CONFIDENCE, ask
    ans = ask(a.question, k=a.k, search_type=a.search, db_path=a.db,
              model=a.model, synthesize=not a.no_synth,
              min_confidence=a.min_confidence or DEFAULT_MIN_CONFIDENCE)
    print(ans.render())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
