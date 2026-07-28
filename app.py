"""Streamlit UI.

The citations are playable links and a terminal cannot click — the UI is what
makes the core feature usable. Everything shown is read from the retrieved rows
or the show manifest; the UI computes nothing about relevance.
"""
from __future__ import annotations

import os

import streamlit as st

# Streamlit does not inherit a sourced shell env, so .env must be loaded here.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from podrag.answer import DEFAULT_MIN_CONFIDENCE, ask
from podrag.index import open_index
from podrag.manifest import load as load_manifest
from podrag.session import Session, standalone_question

st.set_page_config(page_title="podrag", page_icon="🎙", layout="centered")

DB = os.environ.get("PODRAG_DB", "./podrag.lance")

SUGGESTED = [
    "What happens to hormones during menopause and what can help?",
    "How does stress affect mitochondria?",
    "What are the signs someone is manipulating you?",
    "Does exercise help mitochondrial function?",
    "What should women know about hormone therapy?",
]


@st.cache_resource(show_spinner=False)
def _corpus(db_path: str):
    tbl = open_index(db_path)
    return tbl.count_rows(), tbl.version


def _run(question: str, *, k, search_type, floor, synth, model):
    sess: Session = st.session_state.session
    key = os.environ.get("OPENAI_API_KEY")
    q, rewritten = standalone_question(sess, question, model, key)
    a = ask(q, k=k, search_type=search_type, db_path=DB,
            min_confidence=floor, synthesize=synth, model=model)
    sess.add(question, a.text, refused=a.refused)
    st.session_state.last = (question, q if rewritten else None, a)


# ---------------------------------------------------------------- state
if "session" not in st.session_state:
    st.session_state.session = Session()
if "last" not in st.session_state:
    st.session_state.last = None

try:
    n_chunks, version = _corpus(DB)
except Exception as e:
    st.error(f"No index at `{DB}`. Build one:\n\n"
             f"`python -m podrag.cli index <video_id> --show <name>`\n\n{e}")
    st.stop()

manifest = load_manifest()

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.subheader("Corpus")
    c1, c2 = st.columns(2)
    c1.metric("chunks", f"{n_chunks:,}")
    c2.metric("episodes", sum(len(s["episodes"]) for s in manifest.values()))
    st.caption(f"index v{version}")
    st.divider()

    for slug, show in sorted(manifest.items(), key=lambda kv: kv[1]["name"]):
        url = show.get("channel_url") or ""
        st.markdown(f"### [{show['name']}]({url})" if url else f"### {show['name']}")
        for guid, ep in sorted(show["episodes"].items(),
                               key=lambda kv: kv[1].get("published", ""), reverse=True):
            mins = ep.get("duration_s", 0) // 60
            meta = f"{ep.get('published','')[:10]}" + (f" · {mins}m" if mins else "")
            st.markdown(f"[{ep['title'][:58]}]({ep['url']})")
            st.caption(meta)
    st.divider()

    search_type = st.radio("retrieval", ["hybrid", "vector", "keyword"], index=0)
    k = st.slider("passages", 3, 12, 6)
    floor = st.slider("confidence floor", 0.0, 0.6, DEFAULT_MIN_CONFIDENCE, 0.01,
                      help="Below this the system refuses instead of answering. "
                           "This threshold drifts with corpus size — see "
                           "scratch/findings.md F-1.")
    has_key = bool(os.environ.get("OPENAI_API_KEY"))
    synth = st.toggle("synthesize answer", value=has_key, disabled=not has_key,
                      help="Off = retrieval only, no LLM call, no spend."
                           if has_key else "No OPENAI_API_KEY — retrieval only.")
    model = st.selectbox("model", ["gpt-4o-mini", "gpt-4o"], index=0,
                         disabled=not has_key)

    if st.session_state.session.exchanges:
        st.divider()
        st.caption(f"conversation: {len(st.session_state.session.exchanges)} turns")
        if st.button("clear conversation", use_container_width=True):
            st.session_state.session.clear()
            st.session_state.last = None
            st.rerun()

# ---------------------------------------------------------------- main
st.title("podrag")
st.caption("Ask a podcast a question. Every claim links to the second it was said.")

with st.form("q", clear_on_submit=False):
    question = st.text_input("Question", key="qbox",
                             placeholder=SUGGESTED[0])
    submitted = st.form_submit_button("Ask", type="primary")

# Empty submit runs the placeholder, so pressing Enter on an empty box works.
if submitted:
    _run(question.strip() or SUGGESTED[0], k=k, search_type=search_type,
         floor=floor, synth=synth, model=model)

st.caption("Try:")
cols = st.columns(2)
for i, s in enumerate(SUGGESTED[1:]):
    if cols[i % 2].button(s, key=f"sug{i}", use_container_width=True):
        _run(s, k=k, search_type=search_type, floor=floor, synth=synth, model=model)

# ---------------------------------------------------------------- answer
if st.session_state.last:
    asked, rewritten, a = st.session_state.last
    st.divider()
    st.markdown(f"**{asked}**")
    if rewritten:
        st.caption(f"interpreted as: *{rewritten}*")

    if getattr(a, "error", None):
        st.warning(a.error)
    elif a.refused:
        st.warning("**No confident answer.**")
        st.caption(a.reason)
        st.caption("Refused rather than synthesising from weakly-related "
                   "passages — an answer built on irrelevant context is worse "
                   "than no answer.")

    if a.text:
        st.markdown(a.text)

    if a.hits:
        st.divider()
        st.markdown(f"**Sources** · confidence {a.confidence:.3f}")
        for h in a.hits:
            show = manifest.get(h.show, {})
            name = show.get("name", h.show)
            st.markdown(
                f"**{name}** — {h.episode_title}  \n"
                f"[▶ {h.timestamp}]({h.url()})")
            with st.expander("passage"):
                st.caption(h.text)
