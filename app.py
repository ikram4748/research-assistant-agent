"""Streamlit UI: ask a research question, watch the agents work, read a review where every sentence is cited.

    streamlit run app.py
"""
from __future__ import annotations

import json

import streamlit as st

from litreview import pipeline
from litreview.config import ROOT
from litreview.llm import Gemini, LLMUnavailable

st.set_page_config(page_title="Literature Review Agents", page_icon="📚", layout="wide")

STEP = {"plan": "🧭 Planner", "search": "🔎 Searcher", "rank": "📊 Ranking", "read": "📖 Reader", "write": "✍️ Writer",
        "critique": "🧐 Critic", "revise": "🛠️ Reviser"}
EXAMPLES = ["How do vision transformers compare with CNNs on class-imbalanced image classification?",
            "How is retrieval-augmented generation evaluated, and which metrics are used for faithfulness?",
            "What techniques defend LLM applications against prompt injection attacks?",
            "What approaches exist for low-resource Urdu natural language processing?"]


@st.cache_resource
def llm():
    return Gemini()


def describe(node: str, upd: dict) -> str:
    e = (upd.get("events") or [{}])[-1]
    if node == "plan":
        return "Sub-questions:\n" + "\n".join(f"- {x}" for x in e.get("subquestions", []))
    if node == "search":
        return f"{len(e.get('queries', []))} queries to arXiv and OpenAlex → {e.get('found')} papers with abstracts"
    if node == "rank":
        return f"Kept the {e.get('kept')} papers most similar to the question (local embeddings)"
    if node == "read":
        return (f"{e.get('findings')} findings from {e.get('papers_with_findings')} papers. "
                f"**{e.get('dropped_ungrounded')} dropped** because their quote was not in the abstract.")
    if node == "write":
        return f"Draft: {e.get('sentences')} cited sentences ({e.get('invalid_citations')} invalid citations removed)"
    if node == "critique":
        return f"Round {e.get('round')}: {e.get('flagged')} of {e.get('of')} sentences not fully supported" + "".join(
            f"\n- #{d['n']} {d['verdict']}: {d['reason']}" for d in e.get("details", []))
    if node == "revise":
        return f"Rewrote {e.get('rewritten')} sentences, removed {e.get('removed')}"
    return ""


def main() -> None:
    st.title("📚 Literature Review Agents")
    st.caption("Planner → searcher → reader → writer → critic. Every sentence cites a paper, every finding quotes the "
               "paper's abstract word for word, and a critic sends unsupported sentences back for revision.")
    tab_run, tab_eval = st.tabs(["Run", "Evaluation"])
    with tab_eval:
        for split in ("test", "dev"):
            p = ROOT / "results" / f"RESULTS_{split}.md"
            if p.exists():
                st.markdown(p.read_text())
    with tab_run:
        q = st.selectbox("Example questions", ["(write your own)"] + EXAMPLES)
        question = st.text_input("Research question", value="" if q.startswith("(") else q)
        if not st.button("Write review", type="primary", disabled=not question.strip()):
            return
        try:
            graph = pipeline.build(llm())
        except LLMUnavailable as e:
            st.error(str(e))
            return
        state: dict = {}
        left, right = st.columns([2, 3])
        with left, st.status("Agents at work...", expanded=True) as status:
            try:
                for mode, chunk in graph.stream({"question": question}, {"recursion_limit": 40}, stream_mode=["updates", "values"]):
                    if mode == "values":
                        state = chunk
                        continue
                    for node, upd in chunk.items():
                        st.markdown(f"**{STEP.get(node, node)}**  \n" + describe(node, upd))
                status.update(label=f"Done: {state.get('llm_calls', 0)} LLM calls", state="complete")
            except LLMUnavailable as e:
                status.update(label="Gemini quota exhausted", state="error")
                st.error(str(e))
                return
        with right:
            st.markdown(pipeline.render(state))
            st.download_button("Download review (Markdown)", pipeline.render(state), file_name="review.md")
            with st.expander("Raw findings (claim + verbatim quote)"):
                st.code(json.dumps(state.get("findings", {}), indent=1, ensure_ascii=False), language="json")


main()
