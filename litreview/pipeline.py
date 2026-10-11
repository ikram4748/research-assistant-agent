"""The multi-agent literature review, as a LangGraph graph, and the single-agent baseline it is compared with.

    plan -> search -> rank -> read -> write -> critique --all supported / 2 rounds--> finish
                                                 ^   \\--some unsupported--> revise
                                                 \\__________________________/

Deterministic guards (no LLM involved):
  * a finding is kept only if its quote appears word for word in the paper's abstract
  * a citation is kept only if it names a paper the reader actually extracted findings from
  * a sentence left with no valid citation is dropped
"""
from __future__ import annotations

import operator
import re
import time
from functools import lru_cache
from typing import Annotated, TypedDict

import numpy as np
from langgraph.graph import END, StateGraph

from . import prompts, sources
from .config import get_settings
from .llm import Gemini, parse_json

MAX_PAPERS = 12
MAX_ROUNDS = 2
READ_BATCH = 4


class State(TypedDict, total=False):
    question: str
    subquestions: list
    papers: dict                    # id -> paper dict (every search result)
    selected: list                  # paper ids kept after ranking, in order: P1 = selected[0]
    findings: dict                  # P-label -> list of {claim, quote}
    sentences: list                 # [{section, text, cites}]
    verdicts: list
    rounds: int
    events: Annotated[list, operator.add]
    llm_calls: Annotated[int, operator.add]
    dropped: Annotated[dict, lambda a, b: {k: a.get(k, 0) + b.get(k, 0) for k in {*a, *b}}]


def _norm(t: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s%.,\-]", " ", t.lower())).strip()


def quote_in(abstract: str, quote: str) -> bool:
    q = _norm(quote)
    return len(q.split()) >= 4 and q in _norm(abstract)


@lru_cache(maxsize=1)
def _embedder():
    from fastembed import TextEmbedding
    s = get_settings()
    return TextEmbedding(s.embed_model, cache_dir=str(s.cache_dir / "models"))


def rank_papers(texts_for_query: list[str], papers: list[dict], k: int = MAX_PAPERS) -> list[str]:
    """Order papers by the best cosine similarity between their title+abstract and any of the queries."""
    if not papers:
        return []
    e = _embedder()
    q = np.array(list(e.query_embed(texts_for_query)))
    p = np.array(list(e.passage_embed([f"{x['title']}. {x['abstract']}" for x in papers])))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    score = (p @ q.T).max(axis=1)
    return [papers[i]["id"] for i in np.argsort(-score)[:k]]


def label(state_or_selected) -> dict[str, str]:
    sel = state_or_selected["selected"] if isinstance(state_or_selected, dict) else state_or_selected
    return {f"P{i + 1}": pid for i, pid in enumerate(sel)}


def clean_sentences(sections: list[dict], valid: set[str]) -> tuple[list[dict], int, int]:
    """Flatten sections; keep only citations to valid papers; drop sentences left without one."""
    out, bad_cites, uncited = [], 0, 0
    for sec in sections or []:
        for s in sec.get("sentences", []):
            text = INLINE_CITE.sub("", str(s.get("text", ""))).strip()
            if not text:
                continue
            cites = [c for c in s.get("cites", []) if c in valid]
            bad_cites += len(s.get("cites", [])) - len(cites)
            if not cites:
                uncited += 1
                continue
            out.append({"section": sec.get("heading", ""), "text": text, "cites": list(dict.fromkeys(cites))})
    return out, bad_cites, uncited


INLINE_CITE = re.compile(r'\s*\[\s*(?:\\?"?P\d+\\?"?\s*,?\s*)+\]')           # [P1], [P1, P2] in sentence text
BROKEN_CITE = re.compile(r'\s*\[\s*(?:"P\d+"\s*,?\s*)+\](?=")')                    # ["P1"] right before a closing quote


def repair_json(text: str) -> str:
    """Models sometimes repeat the citations inside the sentence text with unescaped quotes ("... design. ["P11"]"),
    which breaks the JSON. Remove such inline citation markers; the "cites" list already holds them."""
    return BROKEN_CITE.sub("", text)


def ask_json(llm, system: str, prompt: str, thinking: str = "low") -> dict:
    """JSON call with one repair pass and one retry: an unparseable reply must not silently become an empty review."""
    r = llm.generate(prompt, system, json_mode=True, thinking=thinking)
    for text in (r.text, repair_json(r.text)):
        try:
            return parse_json(text)
        except ValueError:
            continue
    r = llm.generate(prompt + "\n\nYour previous reply was not valid JSON. Reply again with valid JSON only; put citations "
                     "only in the cites lists, never inside the text.", system, json_mode=True, thinking=thinking)
    try:
        return parse_json(repair_json(r.text))
    except ValueError:
        return {}


def _papers_block(state: dict, ids: list[str], with_findings: bool = False) -> str:
    lab = {v: k for k, v in label(state).items()}
    lines = []
    for pid in ids:
        p = state["papers"][pid]
        head = f"{lab[pid]}: {p['title']} ({p.get('year') or 'n.d.'})"
        if with_findings:
            fs = state["findings"].get(lab[pid], [])
            lines.append(head + "\n" + "\n".join(f"  - {f['claim']} (quote: \"{f['quote']}\")" for f in fs))
        else:
            lines.append(f"{head}\nAbstract: {p['abstract']}")
    return "\n\n".join(lines)


def build(llm: Gemini | None = None):
    llm = llm or Gemini()

    def ask(system: str, prompt: str, thinking: str = "low") -> dict:
        return ask_json(llm, system, prompt, thinking)

    def plan(s: State) -> State:
        out = ask(prompts.PLANNER, f"Research question: {s['question']}")
        subs = [x for x in out.get("subquestions", []) if x.get("text")][:4] or [{"text": s["question"], "queries": [s["question"]]}]
        return {"subquestions": subs, "llm_calls": 1, "rounds": 0, "dropped": {},
                "events": [{"step": "plan", "subquestions": [x["text"] for x in subs]}]}

    def search(s: State) -> State:
        papers = {}
        queries = [q for sub in s["subquestions"] for q in sub.get("queries", [])[:2]]
        for q in queries:
            for p in sources.search(q, n_each=6):
                papers.setdefault(p.id, p.to_dict())
        # the same paper can come back from different queries under the same id; merge by title too
        by_title = {}
        for p in papers.values():
            by_title.setdefault(sources.Paper(**p).key, p)
        papers = {p["id"]: p for p in by_title.values()}
        return {"papers": papers, "events": [{"step": "search", "queries": queries, "found": len(papers)}]}

    def rank(s: State) -> State:
        texts = [s["question"]] + [x["text"] for x in s["subquestions"]]
        sel = rank_papers(texts, list(s["papers"].values()))
        return {"selected": sel, "events": [{"step": "rank", "kept": len(sel)}]}

    def read(s: State) -> State:
        lab = label(s)
        findings, kept, dropped, calls = {}, 0, 0, 0
        for b in range(0, len(s["selected"]), READ_BATCH):
            ids = s["selected"][b:b + READ_BATCH]
            out = ask(prompts.READER.format(question=s["question"]), _papers_block(s, ids))
            calls += 1
            for entry in out.get("papers", []):
                pid = lab.get(str(entry.get("id")))
                if not pid or pid not in ids:
                    continue
                for f in entry.get("findings", [])[:3]:
                    if quote_in(s["papers"][pid]["abstract"], str(f.get("quote", ""))):
                        findings.setdefault(entry["id"], []).append({"claim": str(f["claim"]), "quote": str(f["quote"])})
                        kept += 1
                    else:
                        dropped += 1
        return {"findings": findings, "llm_calls": calls, "dropped": {"ungrounded_findings": dropped},
                "events": [{"step": "read", "findings": kept, "dropped_ungrounded": dropped, "papers_with_findings": len(findings)}]}

    def write(s: State) -> State:
        ids = [label(s)[p] for p in s["findings"]]
        subs = "\n".join(f"- {x['text']}" for x in s["subquestions"])
        out = ask(prompts.WRITER, f"Research question: {s['question']}\n\nSub-questions:\n{subs}\n\nFindings:\n\n"
                  + _papers_block(s, ids, with_findings=True))
        sents, bad, uncited = clean_sentences(out.get("sections", []), set(s["findings"]))
        return {"sentences": sents, "llm_calls": 1, "dropped": {"invalid_citations": bad, "uncited_sentences": uncited},
                "events": [{"step": "write", "sentences": len(sents), "invalid_citations": bad, "uncited": uncited}]}

    def critique(s: State) -> State:
        lab = label(s)
        cited = list(dict.fromkeys(lab[c] for x in s["sentences"] for c in x["cites"]))
        numbered = "\n".join(f"{i + 1}. {x['text']} [cites: {', '.join(x['cites'])}]" for i, x in enumerate(s["sentences"]))
        out = ask(prompts.CRITIC, f"Review sentences:\n{numbered}\n\nCited papers:\n\n{_papers_block(s, cited)}")
        verdicts = [{"n": int(v.get("n", 0)), "verdict": str(v.get("verdict", "")), "reason": str(v.get("reason", ""))}
                    for v in out.get("verdicts", []) if str(v.get("n", "")).isdigit()]
        bad = [v for v in verdicts if v["verdict"] != "supported" and 1 <= v["n"] <= len(s["sentences"])]
        return {"verdicts": verdicts, "rounds": s["rounds"] + 1, "llm_calls": 1,
                "events": [{"step": "critique", "round": s["rounds"] + 1, "flagged": len(bad),
                            "of": len(s["sentences"]), "details": bad}]}

    def revise(s: State) -> State:
        bad = {v["n"]: v for v in s["verdicts"] if v["verdict"] != "supported" and 1 <= v["n"] <= len(s["sentences"])}
        ids = [label(s)[p] for p in s["findings"]]
        listing = "\n".join(f"{n}. {s['sentences'][n - 1]['text']} [cites: {', '.join(s['sentences'][n - 1]['cites'])}] "
                            f"-- problem: {v['verdict']}: {v['reason']}" for n, v in bad.items())
        out = ask(prompts.REVISER, f"Sentences to fix:\n{listing}\n\nFindings:\n\n" + _papers_block(s, ids, with_findings=True))
        sents = [dict(x) for x in s["sentences"]]
        removed = 0
        for r in out.get("revisions", []):
            n = int(r.get("n", 0)) if str(r.get("n", "")).isdigit() else 0
            if n not in bad:
                continue
            text = str(r.get("text", "")).strip()
            cites = [c for c in r.get("cites", []) if c in s["findings"]]
            if not text or not cites:
                sents[n - 1] = None
                removed += 1
            else:
                sents[n - 1] = {**sents[n - 1], "text": text, "cites": cites}
        sents = [x for x in sents if x]
        return {"sentences": sents, "llm_calls": 1,
                "events": [{"step": "revise", "rewritten": len(bad) - removed, "removed": removed}]}

    def after_critique(s: State) -> str:
        flagged = any(v["verdict"] != "supported" for v in s["verdicts"])
        return "revise" if flagged and s["rounds"] < MAX_ROUNDS else END

    g = StateGraph(State)
    for name, fn in [("plan", plan), ("search", search), ("rank", rank), ("read", read), ("write", write),
                     ("critique", critique), ("revise", revise)]:
        g.add_node(name, fn)
    g.set_entry_point("plan")
    for a, b in [("plan", "search"), ("search", "rank"), ("rank", "read"), ("read", "write"), ("write", "critique"),
                 ("revise", "critique")]:
        g.add_edge(a, b)
    g.add_conditional_edges("critique", after_critique, ["revise", END])
    return g.compile()


def run(question: str, llm: Gemini | None = None) -> dict:
    t0 = time.monotonic()
    state = build(llm).invoke({"question": question}, {"recursion_limit": 40})
    state["seconds"] = round(time.monotonic() - t0, 1)
    return state


def baseline(question: str, llm: Gemini | None = None) -> dict:
    """Single agent: one search with the question, the same ranking, one call to write the review. No reader, no critic."""
    llm = llm or Gemini()
    t0 = time.monotonic()
    papers = {p.id: p.to_dict() for p in sources.search(question, n_each=12)}
    sel = rank_papers([question], list(papers.values()))
    state = {"question": question, "papers": papers, "selected": sel}
    out = ask_json(llm, prompts.BASELINE, f"Research question: {question}\n\nPapers:\n\n{_papers_block(state, sel)}")
    sents, bad, uncited = clean_sentences(out.get("sections", []), set(label(state)))
    return {**state, "sentences": sents, "llm_calls": 1, "dropped": {"invalid_citations": bad, "uncited_sentences": uncited},
            "seconds": round(time.monotonic() - t0, 1), "events": [{"step": "baseline", "sentences": len(sents)}]}


def render(state: dict) -> str:
    """Markdown review with numbered references."""
    lab = label(state)
    used = list(dict.fromkeys(c for x in state.get("sentences", []) for c in x["cites"]))
    num = {c: i + 1 for i, c in enumerate(used)}
    blocks: list[tuple[str, list[str]]] = []
    for x in state.get("sentences", []):
        if not blocks or blocks[-1][0] != x.get("section", ""):
            blocks.append((x.get("section", ""), []))
        blocks[-1][1].append(x["text"].rstrip() + " " + "".join(f"[{num[c]}]" for c in x["cites"]))
    body = "\n\n".join((f"### {h}\n\n" if h else "") + " ".join(ss) for h, ss in blocks)
    refs = []
    for c in used:
        p = state["papers"][lab[c]]
        who = (p["authors"][0].split()[-1] + (" et al." if len(p["authors"]) > 1 else "")) if p.get("authors") else "Anon."
        venue = f" *{p['venue']}*." if p.get("venue") else ""
        refs.append(f"[{num[c]}] {who} ({p.get('year') or 'n.d.'}). {p['title'].rstrip('.')}.{venue} {p['url']}")
    return body + "\n\n**References**\n\n" + "\n\n".join(refs)
