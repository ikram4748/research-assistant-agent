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

> **Results: in progress.** The pipeline, baseline, evaluation harness, tests and demo are complete; the 30-question
> evaluation is being run and the table below will be filled from `results/RESULTS_test.md`.

## Why it is built this way

| Choice | Why |
|---|---|
| **Separate critic agent** | A writer checking its own work tends to miss its own mistakes. A separate pass, with only the cited abstracts in front of it, catches sentences the sources do not support. |
| **Verbatim quotes, checked in code** | Every finding must include a quote that appears word for word in the abstract. Findings whose "quote" is not there are dropped before writing: a deterministic guard against invented evidence. |
| **Citations validated in code** | A citation to a paper the reader did not extract findings from is removed, and a sentence left without a valid citation is dropped. |
| **arXiv + OpenAlex** | Both are free and need no key. OpenAlex indexes journals and conferences with abstracts and citation counts. (Semantic Scholar's unauthenticated API rate-limited every request.) Every HTTP response is cached, so runs are reproducible. |
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

*To be filled from `results/RESULTS_test.md`.*

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
