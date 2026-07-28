"""Streamlit chat UI.

Conversational rather than form-based for three reasons, each a defect in the
previous version:
  * st.chat_input submits on Enter natively; a form required clicking a button
  * rendering the exchange history makes follow-ups discoverable — otherwise
    the session memory exists but is invisible and untestable
  * follow-up rewriting is only meaningful if you can see what it rewrote

Everything shown is read from the retrieved rows or the show manifest. The UI
computes nothing about relevance.
"""
from __future__ import annotations

import os

import streamlit as st

# Streamlit does not inherit a sourced shell env.
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
    "What happens to hormones during menopause?",
    "How does stress affect mitochondria?",
    "What are the signs someone is manipulating you?",
    "Does exercise help mitochondrial function?",
]


@st.cache_resource(show_spinner=False)
def _corpus(db_path: str):
    tbl = open_index(db_path)
    return tbl.count_rows(), tbl.version


def _source_line(hit, manifest) -> str:
    """Whole string is the link — a small timestamp is a poor click target."""
    name = manifest.get(hit.show, {}).get("name", hit.show)
    return f"[**{name}** — {hit.episode_title} · ▶ {hit.timestamp}]({hit.url()})"


def _answer(question: str, cfg: dict) -> None:
    sess: Session = st.session_state.session
    key = os.environ.get("OPENAI_API_KEY")
    q, rewritten = standalone_question(sess, question, cfg["model"], key)
    a = ask(q, k=cfg["k"], search_type=cfg["search_type"], db_path=DB,
            min_confidence=cfg["floor"], synthesize=cfg["synth"], model=cfg["model"])
    sess.add(question, a.text, refused=a.refused)
    st.session_state.turns.append(
        {"q": question, "rewritten": q if rewritten else None, "a": a})


# ------------------------------------------------------------------ state
st.session_state.setdefault("session", Session())
st.session_state.setdefault("turns", [])
st.session_state.setdefault("pending", None)

try:
    n_chunks, version = _corpus(DB)
except Exception as e:
    st.error(f"No index at `{DB}`.\n\n`python -m podrag.cli index <video_id> "
             f"--show <name>`\n\n{e}")
    st.stop()

manifest = load_manifest()

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    c1, c2 = st.columns(2)
    c1.metric("chunks", f"{n_chunks:,}")
    c2.metric("episodes", sum(len(s["episodes"]) for s in manifest.values()))
    st.caption(f"index v{version}")

    search_type = st.radio("retrieval", ["hybrid", "vector", "keyword"], index=0,
                           horizontal=True)
    k = st.slider("passages", 3, 12, 6)
    floor = st.slider("confidence floor", 0.0, 0.6, DEFAULT_MIN_CONFIDENCE, 0.01,
                      help="Below this the system refuses. Drifts with corpus "
                           "size — see scratch/findings.md F-1.")
    has_key = bool(os.environ.get("OPENAI_API_KEY"))
    synth = st.toggle("synthesize answer", value=has_key, disabled=not has_key,
                      help="Off = retrieval only, no LLM call, no spend."
                           if has_key else "No OPENAI_API_KEY — retrieval only.")
    if not has_key:
        st.caption("⚠ add `OPENAI_API_KEY` to .env for written answers")
    model = st.selectbox("model", ["gpt-4o-mini", "gpt-4o"], disabled=not has_key)

    if st.session_state.turns and st.button("clear conversation",
                                            use_container_width=True):
        st.session_state.session.clear()
        st.session_state.turns = []
        st.rerun()

    st.divider()
    st.caption("Corpus")
    for slug, show in sorted(manifest.items(), key=lambda kv: kv[1]["name"]):
        url = show.get("channel_url") or ""
        st.markdown(f"**[{show['name']}]({url})**" if url else f"**{show['name']}**")
        for guid, ep in sorted(show["episodes"].items(),
                               key=lambda kv: kv[1].get("published", ""), reverse=True):
            mins = ep.get("duration_s", 0) // 60
            st.markdown(f"<small>[{ep['title'][:52]}]({ep['url']}) "
                        f"· {ep.get('published','')[:10]}"
                        f"{' · ' + str(mins) + 'm' if mins else ''}</small>",
                        unsafe_allow_html=True)

CFG = {"k": k, "search_type": search_type, "floor": floor,
       "synth": synth, "model": model}

# ------------------------------------------------------------------ main
st.title("podrag")
st.caption("Ask a podcast a question. Every claim links to the second it was said. "
           "Follow-ups work — try “what about for men?” after an answer.")

if not st.session_state.turns:
    st.caption("Try one:")
    cols = st.columns(2)
    for i, s in enumerate(SUGGESTED):
        if cols[i % 2].button(s, key=f"sug{i}", use_container_width=True):
            st.session_state.pending = s
            st.rerun()

# render the conversation
for turn in st.session_state.turns:
    with st.chat_message("user"):
        st.markdown(turn["q"])
        if turn["rewritten"]:
            st.caption(f"searched as: *{turn['rewritten']}*")
    with st.chat_message("assistant"):
        a = turn["a"]
        if getattr(a, "error", None):
            st.warning(a.error)
        elif a.refused:
            st.warning("**No confident answer.**")
            st.caption(f"{a.reason} — refused rather than synthesising from "
                       f"weakly-related passages.")
        if a.text:
            st.markdown(a.text)
        if a.hits:
            st.markdown(f"<small>confidence {a.confidence:.3f}</small>",
                        unsafe_allow_html=True)
            for h in a.hits:
                st.markdown(_source_line(h, manifest))
                with st.expander("passage"):
                    st.caption(h.text)

# chat_input submits on Enter
typed = st.chat_input("Ask about the corpus…")
pending = st.session_state.pending or typed
if pending:
    st.session_state.pending = None
    with st.spinner("searching…"):
        _answer(pending, CFG)
    st.rerun()
