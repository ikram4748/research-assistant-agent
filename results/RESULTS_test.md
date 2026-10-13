# Results: literature reviews, test split

Agents: `gemini-3.1-flash-lite`; judge: `gemini-3.5-flash-lite` (a different model, so the multi-agent pipeline is not graded by the critic it was optimised against).

| System | Reviews | Sentences / review | Sentence support | Citation precision | Distinct papers cited | LLM calls / review | Seconds / review |
|---|---|---|---|---|---|---|---|
| `baseline` | 20 | 9.1 | **97%** | 80% | 8.3 | 1.0 | 22 |
| `multi_agent` | 20 | 9.7 | **97%** | 87% | 9.3 | 6.3 | 90 |

**Sentence support, multi-agent − baseline:** -0% per review, 95% CI [-3%, +2%] (paired bootstrap over 20 questions).

**Citation precision, multi-agent − baseline:** +4% per review, 95% CI [-6%, +14%] (paired bootstrap over 20 questions).

**Judge check:** with every citation swapped to a random other retrieved paper, the judge marked 90% of 375 sentences unsupported (its sensitivity to wrong citations).

Inside the multi-agent pipeline: 15 findings were dropped because their quote was not in the abstract, and the critic flagged 3 sentences in its first round (revised or removed before the final review).
