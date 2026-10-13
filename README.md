# Literature Review Agents

Ask a research question; a team of agents searches the scientific literature and writes a short review in which
**every sentence cites a paper, every finding quotes the paper's abstract word for word, and a critic checks each
sentence against its sources** before you see it.

```
question ─► Planner ─► Searcher ─► Ranker ─► Reader ─► Writer ─► Critic ──all supported──► review
            (3-4 sub-   (arXiv +    (local    (findings   (cited     │  ▲
             questions,  OpenAlex,   embed-    + verbatim  review)    │  │ rewrite or remove
             queries)    no key)     dings)    quotes)               ▼  │ (max 2 rounds)
                                                                     Reviser
```

> **Result in one line:** on 20 held-out questions both systems write reviews whose sentences are 97% supported by the
> papers they cite; the multi-agent pipeline cites more precisely (87% vs 80% of citations support their sentence), but
> the paired 95% CI for that gain includes zero, at 6× the LLM calls. The evaluation judge itself had to be fixed first:
> the original one passed everything.

## Why it is built this way

| Choice | Why |
|---|---|
| **Separate critic agent** | A writer checking its own work tends to miss its own mistakes. A separate pass, with only the cited abstracts in front of it, catches sentences the sources do not support. |
| **Verbatim quotes, checked in code** | Every finding must include a quote that appears word for word in the abstract. Findings whose "quote" is not there are dropped before writing: a deterministic guard against invented evidence. |
| **Citations validated in code** | A citation to a paper the reader did not extract findings from is removed, and a sentence left without a valid citation is dropped. |
| **arXiv + OpenAlex** | Both are free and work without a key (OpenAlex now gives keyless users a small shared daily budget; an optional free `OPENALEX_API_KEY` raises it). OpenAlex indexes journals and conferences with abstracts and citation counts. (Semantic Scholar's unauthenticated API rate-limited every request.) Every HTTP response is cached, so runs are reproducible. |
| **Local ranking** | bge-small embeddings (ONNX, CPU) keep the 12 papers most similar to the question and its sub-questions, at no API cost. |
| **Robust JSON** | Models sometimes repeat citations inside the sentence text with unescaped quotes, which breaks the JSON. A repair pass and one retry stop this from silently producing an empty review (this happened during development). |

## Evaluation

30 research questions on ML, NLP, RAG, agents, experimentation and data science (10 dev / 20 test). Each is answered
by the **multi-agent pipeline** and by a **single-agent baseline** (one search with the question, the same ranking,
one call to write the review, no reader and no critic). An **independent judge**, a different model from the agents
with a different prompt from the critic, checks every sentence against the abstracts of the papers it cites:

- **Sentence support:** share of sentences the cited papers support.
- **Citation precision:** share of (sentence, cited paper) pairs where that paper supports the sentence.
- **Judge check:** the same reviews with every citation swapped to a random other retrieved paper should be judged
  unsupported. The share caught is the judge's sensitivity; without it, the scores could not be trusted.

## Results

Test split, 20 questions, one run, models pinned (agents `gemini-3.1-flash-lite`, judge `gemini-3.5-flash-lite`, no
fallback). Full output: [`results/RESULTS_test.md`](results/RESULTS_test.md); every review and judgement is in `results/`.

| System | Sentence support | Citation precision | Sentences / review | Distinct papers cited | LLM calls / review | Seconds / review* |
|---|---|---|---|---|---|---|
| Single-agent baseline | **97%** | 80% | 9.1 | 8.3 | 1.0 | 22 |
| Multi-agent pipeline | **97%** | **87%** | 9.7 | 9.3 | 6.3 | 90 |

\*Wall time on the free tier, including rate-limit waits; not a latency benchmark.

Paired comparison (multi-agent − baseline, per question, bootstrap over the 20 questions):

| Metric | Difference | 95% CI |
|---|---|---|
| Sentence support | −0 points | [−3, +2] |
| Citation precision | +4 points | [−6, +14] |

**Judge check:** with every citation swapped to a random other retrieved paper, the judge marked **90% of 375**
sentences unsupported, so a wrong citation is caught 9 times out of 10.

Inside the multi-agent pipeline, the code-level quote check dropped 15 findings whose "quote" was not in the abstract,
and the critic flagged 3 sentences in its first round (revised or removed before the final review).

### What this means

- **The extra agents did not buy measurably better grounding.** Given the abstracts, one call to a current model already
  writes reviews whose sentences are 97% supported. The multi-agent pipeline cites more precisely on average and cites
  more distinct papers, but at this sample size the precision gain is not distinguishable from zero, and it costs about
  6× the LLM calls.
- **Early numbers were too good.** With 9 of the 20 multi-agent reviews done, multi-agent looked clearly better on
  citation precision (93% vs 80%). With all 20 the gap shrank to +4 points and the CI crossed zero. Reporting only
  after the full held-out run is why the split exists.
- **The dev split agreed:** 98% vs 99% support, 86% vs 86% precision (10 questions,
  [`results/RESULTS_dev.md`](results/RESULTS_dev.md)).

### The judge that passed everything

The first judge (a "is this sentence supported?" prompt) gave both systems **100%** support on dev. The judge check
exposed it: with citations swapped to wrong papers it still passed half of them (sensitivity **48%** on 182
sentences). Reading its output showed why: all sentences of a review are judged in one call, and the judge credited a
sentence with *another* sentence's paper. Two fixes, both before the test run:

1. The judge prompt became a fact-checker: list the sentence's claims, quote the words in the cited abstract that
   support them, and no quote means no support.
2. A code check: a sentence counts as supported only if the judge names a paper the sentence actually cites **and** its
   quoted evidence is found in that paper's abstract (fuzzy match: a 4+ word fragment, 80% verbatim).

Sensitivity rose to 89% on dev and 90% on test. One more failure surfaced on test: the judge's reply for one question
contained a LaTeX backslash copied from an abstract (`\(`), which is invalid JSON, so all 9 sentences were silently
left unjudged. The JSON parser now repairs invalid escapes, and that question was re-judged from the cached reply.

### Limitations

- 20 test questions: CIs are wide; a precision difference of a few points cannot be resolved.
- The judge sees abstracts only, not full papers, and it is an LLM: 10% of wrong citations still pass.
- One run per system; reviews depend on what arXiv and OpenAlex returned on the day (responses are cached for
  reproducibility).

## Run it

**Windows:** double-click `run.bat`. **macOS:** double-click `run.command`. The first run installs packages, asks for
a free Gemini API key and opens the app, where you can watch each agent step.

```bash
pip install -r requirements.txt
cp .env.example .env                        # add GEMINI_API_KEY
streamlit run app.py
python -m evaluation.run_eval --split test  # reviews + judge for both systems
python -m pytest                            # offline tests (fake LLM, fake search)
```

## Project layout

```
litreview/
  pipeline.py   LangGraph graph (plan, search, rank, read, write, critique, revise), baseline, rendering
  sources.py    arXiv and OpenAlex search, merged by title, cached
  prompts.py    every prompt
  llm.py        Gemini client: two keys, strict model preference, rate limiting, local cache
evaluation/     questions.yaml, run_eval.py (reviews, judge, judge check, report)
app.py          Streamlit UI
tests/          offline tests
```

## Stack

Python · LangGraph · Gemini (`google-genai`) · arXiv API · OpenAlex API · fastembed (bge-small, ONNX) · Streamlit · pytest

## License

MIT
