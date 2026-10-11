"""Citation-grounding evaluation: multi-agent pipeline vs single-agent baseline.

    python -m evaluation.run_eval                    # test split, both systems, then the judge
    python -m evaluation.run_eval --split dev
    python -m evaluation.run_eval --report

For every review, an independent judge (a different model from the agents, with a different prompt from the
critic) checks each sentence against the abstracts of the papers it cites:
    sentence support     share of sentences the cited papers support
    citation precision   share of (sentence, cited paper) pairs where that paper supports the sentence
plus counts that need no judge: sentences, distinct papers cited, invalid citations removed, ungrounded findings.

Judge check: the same reviews with every citation swapped to a random other retrieved paper should be judged
unsupported. The share caught is the judge's sensitivity; without this check the scores could not be trusted.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from litreview import pipeline, prompts  # noqa: E402
from litreview.config import get_settings  # noqa: E402
from litreview.llm import Gemini, LLMUnavailable, parse_json  # noqa: E402

RESULTS = ROOT / "results"
SYSTEMS = ["baseline", "multi_agent"]


def questions(split: str) -> list[dict]:
    rows = yaml.safe_load((ROOT / "evaluation" / "questions.yaml").read_text())
    return [r for r in rows if split == "all" or r["split"] == split]


def _jsonl(p: Path) -> dict:
    return {r["id"]: r for r in map(json.loads, p.read_text().splitlines())} if p.exists() else {}


def judge(llm: Gemini, review: dict, cites_override: list[list[str]] | None = None) -> list[dict]:
    lab = pipeline.label(review["selected"])
    sents = review["sentences"]
    blocks = []
    for i, s in enumerate(sents):
        cites = cites_override[i] if cites_override else s["cites"]
        abstracts = "\n".join(f"  {c}: {review['papers'][lab[c]]['title']}. {review['papers'][lab[c]]['abstract']}" for c in cites)
        blocks.append(f"{i + 1}. {s['text']}\n Cited papers:\n{abstracts}")
    r = llm.generate("\n\n".join(blocks), prompts.JUDGE, json_mode=True, thinking="low")
    try:
        out = {int(x["n"]): x for x in parse_json(r.text).get("sentences", []) if str(x.get("n", "")).isdigit()}
    except ValueError:
        out = {}
    res = []
    for i, s in enumerate(sents):
        cites = cites_override[i] if cites_override else s["cites"]
        v = out.get(i + 1, {})
        res.append({"supported": bool(v.get("supported")), "supporting": [c for c in v.get("supporting", []) if c in cites],
                    "cites": cites, "judged": bool(v)})
    return res


def produce(split: str, system: str, llm: Gemini, workers: int) -> None:
    path = RESULTS / "reviews" / split / f"{system}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    done, lock = _jsonl(path), threading.Lock()
    todo = [q for q in questions(split) if q["id"] not in done]
    print(f"{system}: {len(done)} done, {len(todo)} to go")

    def one(q):
        s = pipeline.run(q["question"], llm) if system == "multi_agent" else pipeline.baseline(q["question"], llm)
        row = {"id": q["id"], "question": q["question"], "system": system, "selected": s["selected"],
               "papers": {pid: s["papers"][pid] for pid in s["selected"]}, "sentences": s["sentences"],
               "llm_calls": s["llm_calls"], "seconds": s["seconds"], "dropped": s.get("dropped", {}),
               "events": s["events"], "markdown": pipeline.render(s)}
        with lock, path.open("a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"  {system} {q['id']}: {len(s['sentences'])} sentences, {s['llm_calls']} calls, {s['seconds']}s")

    with ThreadPoolExecutor(workers) as pool:
        for f in [pool.submit(one, q) for q in todo]:
            try:
                f.result()
            except LLMUnavailable as e:
                print("stopping:", e)
                pool.shutdown(cancel_futures=True)
                return


def run_judge(split: str, judge_llm: Gemini) -> None:
    for system in SYSTEMS:
        reviews = _jsonl(RESULTS / "reviews" / split / f"{system}.jsonl")
        path = RESULTS / "judgements" / split / f"{system}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        done = _jsonl(path)
        for qid, rev in reviews.items():
            if qid in done or not rev["sentences"]:
                continue
            rng = random.Random(qid)
            labels = list(pipeline.label(rev["selected"]))
            swapped = [[rng.choice([lab for lab in labels if lab not in s["cites"]] or labels) for _ in s["cites"]]
                       for s in rev["sentences"]]
            try:
                real, fake = judge(judge_llm, rev), judge(judge_llm, rev, swapped)
            except LLMUnavailable as e:
                print("stopping:", e)
                return
            with path.open("a") as f:
                f.write(json.dumps({"id": qid, "system": system, "real": real, "swapped": fake}) + "\n")
            print(f"  judged {system} {qid}")


def report(split: str) -> str:
    lines = [f"# Results: literature reviews, {split} split", "",
             f"Agents: `{get_settings().models[0]}`; judge: `{get_settings().judge_models[0]}` (a different model, so the "
             "multi-agent pipeline is not graded by the critic it was optimised against).", "",
             "| System | Reviews | Sentences / review | Sentence support | Citation precision | Distinct papers cited | "
             "LLM calls / review | Seconds / review |", "|---|---|---|---|---|---|---|---|"]
    per_q = {}
    check = {"swapped_flagged": 0, "swapped_total": 0}
    for system in SYSTEMS:
        revs = _jsonl(RESULTS / "reviews" / split / f"{system}.jsonl")
        judg = _jsonl(RESULTS / "judgements" / split / f"{system}.jsonl")
        if not revs:
            continue
        n_s = sup = pairs = good_pairs = 0
        for qid, j in judg.items():
            js = [x for x in j["real"] if x["judged"]]
            n_s += len(js)
            sup += sum(x["supported"] for x in js)
            pairs += sum(len(x["cites"]) for x in js)
            good_pairs += sum(len(x["supporting"]) for x in js)
            per_q.setdefault(qid, {})[system] = sum(x["supported"] for x in js) / len(js) if js else None
            fk = [x for x in j["swapped"] if x["judged"]]
            check["swapped_total"] += len(fk)
            check["swapped_flagged"] += sum(not x["supported"] for x in fk)
        r = list(revs.values())
        distinct = sum(len({c for s in x["sentences"] for c in s["cites"]}) for x in r) / len(r)
        lines.append(f"| `{system}` | {len(r)} | {sum(len(x['sentences']) for x in r) / len(r):.1f} | "
                     f"**{sup / n_s:.0%}** | {good_pairs / pairs:.0%} | {distinct:.1f} | "
                     f"{sum(x['llm_calls'] for x in r) / len(r):.1f} | {sum(x['seconds'] for x in r) / len(r):.0f} |"
                     if n_s else f"| `{system}` | {len(r)} | - | (not judged yet) | | | | |")
    both = [(v["baseline"], v["multi_agent"]) for v in per_q.values() if v.get("baseline") is not None and v.get("multi_agent") is not None]
    if both:
        diffs = [b - a for a, b in both]
        rng = random.Random(0)
        boots = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(10_000))
        lines += ["", f"**Sentence support, multi-agent − baseline:** {sum(diffs) / len(diffs):+.0%} per review, "
                  f"95% CI [{boots[250]:+.0%}, {boots[9750]:+.0%}] (paired bootstrap over {len(diffs)} questions)."]
    if check["swapped_total"]:
        lines += ["", f"**Judge check:** with every citation swapped to a random other retrieved paper, the judge marked "
                  f"{check['swapped_flagged'] / check['swapped_total']:.0%} of {check['swapped_total']} sentences unsupported "
                  "(its sensitivity to wrong citations)."]
    ma = _jsonl(RESULTS / "reviews" / split / "multi_agent.jsonl")
    if ma:
        drops = sum(x["dropped"].get("ungrounded_findings", 0) for x in ma.values())
        flagged = sum(e.get("flagged", 0) for x in ma.values() for e in x["events"] if e["step"] == "critique" and e["round"] == 1)
        lines += ["", f"Inside the multi-agent pipeline: {drops} findings were dropped because their quote was not in the abstract, "
                  f"and the critic flagged {flagged} sentences in its first round (revised or removed before the final review)."]
    text = "\n".join(lines) + "\n"
    (RESULTS / f"RESULTS_{split}.md").write_text(text)
    return text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test"], default="test")
    ap.add_argument("--systems", nargs="+", default=SYSTEMS, choices=SYSTEMS)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    s = get_settings()
    if not args.report:
        llm = Gemini(dataclasses.replace(s, models=s.models[:1]))           # pinned: no fallback during evaluation
        judge_llm = Gemini(dataclasses.replace(s, models=s.judge_models[:1]))
        for system in args.systems:
            produce(args.split, system, llm, args.workers)
        run_judge(args.split, judge_llm)
    print(report(args.split))


if __name__ == "__main__":
    main()
