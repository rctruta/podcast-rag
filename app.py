"""Streamlit UI.

Exists for one reason: the citations are playable links, and a terminal cannot
click. Everything shown here is read from the retrieved rows — the UI does not
compute or embellish anything, so what you see is what the pipeline stored.

Run:  streamlit run app.py
"""
from __future__ import annotations

import os

import streamlit as st

# Streamlit does not inherit a sourced shell env, so .env must be loaded here.
# Without this, synthesis died with KeyError: 'OPENAI_API_KEY'.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from podrag.answer import DEFAULT_MIN_CONFIDENCE, ask
from podrag.index import open_index

st.set_page_config(page_title="podrag", page_icon="🎙", layout="centered")


@st.cache_resource(show_spinner=False)
def corpus_stats(db_path: str):
    tbl = open_index(db_path)
    rows = tbl.search().limit(100_000).to_list()
    eps, shows = {}, {}
    for r in rows:
        eps[r["episode_guid"]] = r["episode_title"]
        shows.setdefault(r["show"], set()).add(r["episode_guid"])
    return len(rows), eps, shows, tbl.version


DB = os.environ.get("PODRAG_DB", "./podrag.lance")

st.title("podrag")
st.caption("Ask a podcast a question. Every claim links to the second it was said.")

try:
    n_chunks, episodes, shows, version = corpus_stats(DB)
except Exception as e:
    st.error(f"No index at `{DB}`. Build one first:\n\n"
             f"`python -m podrag.cli index <video_id> --show <name>`\n\n{e}")
    st.stop()

with st.sidebar:
    st.subheader("Corpus")
    st.metric("chunks", f"{n_chunks:,}")
    st.metric("episodes", len(episodes))
    st.caption(f"index version {version}")
    st.divider()
    for show, guids in sorted(shows.items()):
        st.markdown(f"**{show}** · {len(guids)}")
        for g in guids:
            st.caption(f"· {episodes[g][:46]}")
    st.divider()
    search_type = st.radio("retrieval", ["hybrid", "vector", "keyword"], index=0)
    k = st.slider("passages", 3, 12, 6)
    floor = st.slider("confidence floor", 0.0, 0.6, DEFAULT_MIN_CONFIDENCE, 0.01,
                      help="Below this the system refuses instead of answering. "
                           "Note: this threshold drifts with corpus size — see "
                           "scratch/findings.md F-1.")
    has_key = bool(os.environ.get("OPENAI_API_KEY"))
    synth = st.toggle("synthesize answer", value=has_key, disabled=not has_key,
                      help="Off = retrieval only, no LLM call, no spend."
                           if has_key else
                           "No OPENAI_API_KEY found. Retrieval still works; "
                           "put a key in .env to enable synthesis.")
    if not has_key:
        st.caption("⚠ no `OPENAI_API_KEY` — retrieval only")

q = st.text_input("Question", placeholder="what happens to hormones during menopause?")

if q:
    with st.spinner("searching…"):
        a = ask(q, k=k, search_type=search_type, db_path=DB,
                min_confidence=floor, synthesize=synth)

    if a.error:
        st.error(a.error)
    elif a.refused:
        st.warning("**No confident answer.**")
        st.caption(a.reason)
        st.progress(min(a.confidence / max(floor, 1e-6), 1.0))
        st.caption(
            "The system refused rather than synthesising from weakly-related "
            "passages. This is the intended behaviour — an answer built on "
            "irrelevant context is worse than no answer.")
    else:
        if synth:
            st.markdown(a.text)
        st.divider()
        st.markdown(f"**Sources** · confidence {a.confidence:.3f}")
        for h in a.hits:
            st.markdown(f"[{h.episode_title[:70]} · **{h.timestamp}**]({h.url()})")
            with st.expander("passage"):
                st.caption(h.text)
