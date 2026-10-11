"""Offline tests: fake LLM and fake search, no network."""
from __future__ import annotations

import json

import pytest

from litreview import pipeline, sources
from litreview.llm import LLMResult


ABS = ("We study long-tailed recognition with vision transformers. Our experiments show that ViTs degrade more than "
       "CNNs on imbalanced data, and that loss reweighting recovers most of the gap on ImageNet-LT.")


def paper(i, abstract=ABS, title=None):
    return sources.Paper(f"openalex:W{i}", title or f"Paper {i} on transformers", abstract, 2024, ["A. Author", "B. Author"],
                         "Some Venue", f"https://doi.org/10/{i}", 3, "openalex")


def test_quote_must_be_verbatim():
    assert pipeline.quote_in(ABS, "ViTs degrade more than CNNs on imbalanced data")
    assert pipeline.quote_in(ABS, "vits   DEGRADE more than cnns on imbalanced data,")     # case and spacing
    assert not pipeline.quote_in(ABS, "ViTs degrade much more than CNNs")                    # paraphrase
    assert not pipeline.quote_in(ABS, "imbalanced data")                                      # too short to mean anything


def test_invalid_and_missing_citations_are_removed():
    sections = [{"heading": "H", "sentences": [{"text": "A.", "cites": ["P1", "P9"]}, {"text": "B.", "cites": ["P9"]},
                                                {"text": "C.", "cites": []}]}]
    out, bad, uncited = pipeline.clean_sentences(sections, {"P1", "P2"})
    assert [s["text"] for s in out] == ["A."] and out[0]["cites"] == ["P1"] and bad == 2 and uncited == 2


def test_openalex_abstract_rebuilt_from_inverted_index():
    assert sources._abstract_from_index({"hello": [0], "world": [1, 3], "big": [2]}) == "hello world big world"


def test_render_numbers_references_in_order_of_use():
    st = {"selected": ["openalex:W1", "openalex:W2"], "papers": {p.id: p.to_dict() for p in (paper(1), paper(2))},
          "sentences": [{"section": "S", "text": "X.", "cites": ["P2"]}, {"section": "S", "text": "Y.", "cites": ["P1", "P2"]}]}
    md = pipeline.render(st)
    assert "X. [1] Y. [2][1]" in md and md.index("Paper 2") < md.index("Paper 1")


class FakeLLM:
    """Answers each agent from a script keyed by a word in its system prompt."""

    def __init__(self, script):
        self.script, self.calls = script, []

    def generate(self, prompt, system="", **kw):
        for key, replies in self.script.items():
            if key in system:
                self.calls.append(key)
                return LLMResult(json.dumps(replies.pop(0)), "fake", False, 0.0)
        raise AssertionError(f"unexpected prompt: {system[:60]}")


@pytest.fixture
def offline(monkeypatch):
    monkeypatch.setattr(sources, "search", lambda q, n_each=8: [paper(1), paper(2, title="Unrelated cooking paper")])
    monkeypatch.setattr(pipeline, "rank_papers", lambda texts, papers, k=12: [p["id"] for p in papers][:k])


def test_pipeline_drops_fake_quotes_and_revises_flagged_sentences(offline):
    llm = FakeLLM({
        "plan a short literature review": [{"subquestions": [{"text": "How do ViTs do?", "queries": ["vit imbalance"]}]}],
        "extract findings": [{"papers": [
            {"id": "P1", "findings": [{"claim": "ViTs degrade more.", "quote": "ViTs degrade more than CNNs on imbalanced data"},
                                      {"claim": "Made up.", "quote": "transformers solve every imbalance problem completely"}]},
            {"id": "P2", "findings": []}]}],
        "write a concise literature review": [{"sections": [{"heading": "ViTs", "sentences": [
            {"text": "ViTs degrade more than CNNs under imbalance.", "cites": ["P1"]},
            {"text": "ViTs always win.", "cites": ["P1", "P7"]}]}]}],
        "check a literature review": [{"verdicts": [{"n": 1, "verdict": "supported"}, {"n": 2, "verdict": "unsupported", "reason": "x"}]},
                                      {"verdicts": [{"n": 1, "verdict": "supported"}]}],
        "not fully supported": [{"revisions": [{"n": 2, "text": "", "cites": []}]}],
    })
    s = pipeline.run("ViT vs CNN on imbalanced data?", llm)
    assert s["dropped"]["ungrounded_findings"] == 1 and s["dropped"]["invalid_citations"] == 1
    assert [x["text"] for x in s["sentences"]] == ["ViTs degrade more than CNNs under imbalance."]
    assert s["rounds"] == 2 and llm.calls.count("check a literature review") == 2
    assert "P2" not in s["findings"]                       # papers without findings cannot be cited


def test_revise_loop_is_bounded(offline):
    flagged = {"verdicts": [{"n": 1, "verdict": "partial", "reason": "x"}]}
    llm = FakeLLM({
        "plan a short literature review": [{"subquestions": [{"text": "q", "queries": ["q"]}]}],
        "extract findings": [{"papers": [{"id": "P1", "findings": [{"claim": "c", "quote": "loss reweighting recovers most of the gap"}]}]}],
        "write a concise literature review": [{"sections": [{"heading": "h", "sentences": [{"text": "t", "cites": ["P1"]}]}]}],
        "check a literature review": [flagged, flagged, flagged],
        "not fully supported": [{"revisions": [{"n": 1, "text": "t2", "cites": ["P1"]}]}] * 3,
    })
    s = pipeline.run("q?", llm)
    assert s["rounds"] == pipeline.MAX_ROUNDS and llm.calls.count("check a literature review") == pipeline.MAX_ROUNDS


def test_inline_citations_with_unescaped_quotes_are_repaired():
    # regression: the writer once returned "text": "... design. ["P11"]", which is invalid JSON and emptied a review
    bad = '{"sections":[{"heading":"h","sentences":[{"text":"Design matters. ["P11"]","cites":["P11"]}]}]}'
    from litreview.llm import parse_json
    d = parse_json(pipeline.repair_json(bad))
    sents, _, _ = pipeline.clean_sentences(d["sections"], {"P11"})
    assert sents == [{"section": "h", "text": "Design matters.", "cites": ["P11"]}]
    assert pipeline.clean_sentences([{"sentences": [{"text": "x [P1, P2]", "cites": ["P1"]}]}], {"P1"})[0][0]["text"] == "x"
