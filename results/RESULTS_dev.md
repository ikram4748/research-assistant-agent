# Results: literature reviews, dev split

Agents: `gemini-3.1-flash-lite`; judge: `gemini-3.5-flash-lite` (a different model, so the multi-agent pipeline is not graded by the critic it was optimised against).

| System | Reviews | Sentences / review | Sentence support | Citation precision | Distinct papers cited | LLM calls / review | Seconds / review |
|---|---|---|---|---|---|---|---|
| `baseline` | 10 | 8.9 | **98%** | 86% | 8.1 | 1.0 | 10 |
| `multi_agent` | 10 | 9.3 | **99%** | 86% | 8.6 | 6.2 | 10 |

**Sentence support, multi-agent − baseline:** +1% per review, 95% CI [-3%, +5%] (paired bootstrap over 10 questions).

**Citation precision, multi-agent − baseline:** -1% per review, 95% CI [-15%, +11%] (paired bootstrap over 10 questions).

**Judge check:** with every citation swapped to a random other retrieved paper, the judge marked 89% of 182 sentences unsupported (its sensitivity to wrong citations).

Inside the multi-agent pipeline: 6 findings were dropped because their quote was not in the abstract, and the critic flagged 1 sentences in its first round (revised or removed before the final review).
