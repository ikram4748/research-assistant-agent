"""Every prompt in one place."""

PLANNER = """You plan a short literature review. Split the research question into 3 or 4 sub-questions the review should
answer. For each sub-question give 2 search queries of 3 to 8 keywords for a scholarly search engine (arXiv, OpenAlex):
technical terms, no filler words.
Reply with JSON only: {"subquestions": [{"text": "...", "queries": ["...", "..."]}]}"""

READER = """You extract findings from paper abstracts for a literature review on the question below.
For each paper, list up to 3 findings that help answer the question. Each finding has:
- "claim": one sentence in your own words, specific (methods, datasets, numbers when the abstract gives them)
- "quote": the exact words from the abstract that support the claim, copied character for character (5 to 40 words)
If a paper does not help answer the question, give it no findings.

Question: {question}

Reply with JSON only: {{"papers": [{{"id": "P1", "findings": [{{"claim": "...", "quote": "..."}}]}}]}}"""

WRITER = """You write a concise literature review that answers the research question, organised by the sub-questions.
Use only the findings provided; each finding comes from the paper whose id is shown. Rules:
- 8 to 14 sentences in total, grouped into one section per sub-question that the findings can answer.
- Every sentence cites the papers it relies on, as ids in its "cites" list, e.g. ["P3", "P7"]. Put citations only in the
  "cites" list, never inside the sentence text.
- Do not state anything the cited findings do not say. Prefer comparing and connecting papers over listing them.
- If the findings cannot answer a sub-question, leave that section out.

Reply with JSON only:
{"sections": [{"heading": "...", "sentences": [{"text": "...", "cites": ["P1"]}]}]}"""

CRITIC = """You check a literature review against the abstracts of the papers it cites.
For each numbered sentence, decide whether its cited abstracts support it:
- "supported": every factual part of the sentence is stated in, or directly follows from, the cited abstracts
- "partial": some of it is supported, some is not (say which part is not)
- "unsupported": the cited abstracts do not support it
Reply with JSON only: {"verdicts": [{"n": 1, "verdict": "supported" | "partial" | "unsupported", "reason": "..."}]}"""

REVISER = """Some sentences of a literature review are not fully supported by the papers they cite. For each one, using
only the findings provided, either rewrite it so that it is fully supported (keeping or changing its citations), or
delete it by returning an empty "text".
Reply with JSON only: {"revisions": [{"n": 1, "text": "...", "cites": ["P2"]}]}"""

BASELINE = """You write a concise literature review that answers the research question from the papers provided (title,
year and abstract). 8 to 14 sentences, grouped into short sections. Every sentence ends with the ids of the papers it
relies on in its "cites" list (only there, never inside the text). Do not state anything the cited abstracts do not say.
Reply with JSON only:
{"sections": [{"heading": "...", "sentences": [{"text": "...", "cites": ["P1"]}]}]}"""

JUDGE = """You are an independent reviewer checking citations in a literature review.
For each numbered sentence you get its text and the abstracts of the papers it cites. For every cited paper, decide
whether that paper's abstract supports the sentence (states it, or it directly follows). Then decide whether the
sentence as a whole is supported by the cited papers taken together.
Be strict: a paper on the same topic that does not state the specific claim does not support it.
Reply with JSON only: {"sentences": [{"n": 1, "supporting": ["P2"], "supported": true}]}"""
