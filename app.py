"""Streamlit chat UI — canonical pattern, no invention.

Layout is Streamlit's documented chat idiom: history renders oldest-first with
st.chat_message, and st.chat_input is pinned to the bottom of the viewport.
Earlier attempts here moved the input to the top and reversed the history;
both were departures from the standard and both were worse.

Everything displayed comes from the retrieved rows or the show manifest. The
UI computes nothing about relevance.
"""
from __future__ import annotations

import os

import streamlit as st

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
    name = manifest.get(hit.show, {}).get("name", hit.show)
    return f"[**{name}** — {hit.episode_title} · ▶ {hit.timestamp}]({hit.url()})"


st.session_state.setdefault("session", Session())
st.session_state.setdefault("turns", [])
st.session_state.setdefault("pending", None)

try:
    n_chunks, version = _corpus(DB)
except Exception as e:
    st.error(f"No index at `{DB}`.\n\n"
             f"`python -m podrag.cli index <video_id> --show <name>`\n\n{e}")
    st.stop()

manifest = load_manifest()

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    c1, c2 = st.columns(2)
    c1.metric("chunks", f"{n_chunks:,}")
    c2.metric("episodes", sum(len(s["episodes"]) for s in manifest.values()))
    st.caption(f"index v{version}")

    search_type = st.radio("retrieval", ["hybrid", "vector", "keyword"],
                           horizontal=True)
    k = st.slider("passages", 3, 12, 6)
    floor = st.slider("confidence floor", 0.0, 0.6, DEFAULT_MIN_CONFIDENCE, 0.01,
                      help="Below this the system refuses. Drifts with corpus "
                           "size — see scratch/findings.md F-1.")
    has_key = bool(os.environ.get("OPENAI_API_KEY"))
    synth = st.toggle("synthesize answer", value=has_key, disabled=not has_key)
    if not has_key:
        st.caption("⚠ add `OPENAI_API_KEY` to .env for written answers")
    model = st.selectbox("model", ["gpt-4o-mini", "gpt-4o"], disabled=not has_key)

    total_calls = sum(t.get("calls", 0) for t in st.session_state.turns)
    if st.session_state.turns:
        st.caption(f"OpenAI calls this session: **{total_calls}**")
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
                               key=lambda kv: kv[1].get("published", ""),
                               reverse=True):
            mins = ep.get("duration_s", 0) // 60
            st.markdown(
                f"<small>[{ep['title'][:50]}]({ep['url']}) · "
                f"{ep.get('published','')[:10]}"
                f"{' · ' + str(mins) + 'm' if mins else ''}</small>",
                unsafe_allow_html=True)

CFG = {"k": k, "search_type": search_type, "floor": floor,
       "synth": synth, "model": model}

# ---------------------------------------------------------------- main
st.title("podrag")
st.caption("Every claim links to the second it was said. Follow-ups work — "
           "ask “what about for men?” after an answer.")

# Starting a new line of questioning must be one obvious click, not a hunt
# through the sidebar. Without it, every question inherits prior context.
if st.session_state.turns:
    left, right = st.columns([3, 1])
    left.caption(f"{len(st.session_state.turns)} question(s) in this thread — "
                 f"follow-ups use the earlier context")
    if right.button("New topic", use_container_width=True, type="secondary"):
        st.session_state.session.clear()
        st.session_state.turns = []
        st.rerun()

# suggestions only before the conversation starts
if not st.session_state.turns:
    cols = st.columns(2)
    for i, sug in enumerate(SUGGESTED):
        if cols[i % 2].button(sug, key=f"sug{i}", use_container_width=True):
            st.session_state.pending = sug
            st.rerun()


def _render(turn):
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
            calls = turn.get("calls", 0)
            cost = "no API calls — retrieval is local" if calls == 0 else \
                   f"{calls} OpenAI call{'s' if calls > 1 else ''}"
            st.caption(f"confidence {a.confidence:.3f} · {cost}")
            for h in a.hits:
                st.markdown(_source_line(h, manifest))
                with st.expander("passage"):
                    st.caption(h.text)


# history: oldest first
for turn in st.session_state.turns:
    _render(turn)

# input: pinned to the bottom of the viewport, submits on Enter
typed = st.chat_input("Ask about the corpus…")
question = st.session_state.pending or typed
st.session_state.pending = None

if question:
    # Compute only — do NOT render here. History below is the single render
    # path. Rendering inline AND re-rendering after st.rerun() produced two
    # assistant blocks, the first partial (text + confidence, no citations),
    # which read as "the first answer has no provenance".
    sess: Session = st.session_state.session
    key = os.environ.get("OPENAI_API_KEY")
    with st.spinner("searching…"):
        q, rewritten = standalone_question(sess, question, model, key)
        a = ask(q, k=k, search_type=search_type, db_path=DB,
                min_confidence=floor, synthesize=synth, model=model)
    sess.add(question, a.text, refused=a.refused)
    st.session_state.turns.append(
        {"q": question, "rewritten": q if rewritten else None, "a": a,
         "calls": (1 if rewritten else 0) + (1 if (synth and a.text and not a.refused) else 0)})
    st.rerun()
