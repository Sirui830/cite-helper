"""Regression tests for retrieval quality.

Each test runs a real query against a real index and asserts:
  (a) the expected paper appears in top-k
  (b) no result comes from a noise section
  (c) no obvious reference-list entries appear

Requires a built index at INDEX_FOLDER (defaults to /tmp/cite-helper-test).
Run with: pytest -xvs tests/test_regression.py
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from cite_helper.chunker import is_retrieval_noise
from cite_helper.retriever import Index


INDEX_FOLDER = Path(
    os.environ.get("CITE_HELPER_TEST_FOLDER", "/tmp/cite-helper-test")
)


@pytest.fixture(scope="module")
def idx() -> Index:
    if not (INDEX_FOLDER / ".cite_helper_index").exists():
        pytest.skip(
            f"No index at {INDEX_FOLDER}. Build with: "
            f"cite-helper build {INDEX_FOLDER} --from-source-index "
            f"/path/to/source_index.jsonl"
        )
    return Index(INDEX_FOLDER)


# (query, citation_key_substring_or_None, k)
# When citation_key_substring is set, that substring must appear in some
# top-k result's citation_key. None means we only assert no noise.
QUERIES = [
    ("politeness strategies vary across cultures", None, 5),
    ("Korean learners use honorifics from their L1", "Yang", 5),
    ("self-praise on Chinese Weibo", "Ren_Guo", 5),
    ("WeChat users prefer direct requests", "Liu_et_al", 5),
    ("rapport management framework", "Spencer_Oatey", 5),
    ("Chinese and Japanese internal vs external modifiers", "Fukushima", 5),
    ("apology strategies in second language learners", "Yang", 8),
    ("face-threatening acts in computer-mediated communication",
     "Flores_Salgado", 8),
    ("modesty maxim in self-praise", "Ren", 5),
    ("study abroad and pragmatic development", "Ren2019", 8),
    ("longitudinal study of L2 request acquisition", None, 5),
    ("discourse markers in online forums", "Elena_Landone", 5),
    ("politeness in workplace email", None, 5),
    ("learner identity and language socialization", None, 5),
    ("conventional indirectness in requests", None, 8),
    ("WhatsApp messages politeness", "Flores_Salgado", 5),
    ("internal modifiers in Japanese requests", "Fukushima", 5),
    ("intercultural communication and face", None, 8),
    ("computational politeness classification", "Priya", 5),
    ("speech act theory and politeness", None, 8),
    ("L2 email request strategies", "Qin", 8),
]


NOISE_SECTION_KEYS = {
    "references",
    "bibliography",
    "acknowledgments",
    "acknowledgements",
    "funding",
    "keywords",
}


def _is_reference_like(sentence: str) -> bool:
    import re
    if re.search(r"\b(19|20)\d{2}[a-z]?\b.*\.\s*[A-Z]", sentence) and (
        sentence.count(",") >= 3 or re.search(r"\bpp?\.\s*\d", sentence)
    ):
        return True
    return False


@pytest.mark.parametrize("query,expected_paper,k", QUERIES)
def test_query_returns_clean_results(
    idx: Index, query: str, expected_paper: str | None, k: int
) -> None:
    hits = idx.find(query, k=k)
    assert len(hits) > 0, f"no hits for: {query!r}"

    for hit in hits:
        section_norm = hit.sentence.section.lower().strip(" :.")
        assert section_norm not in NOISE_SECTION_KEYS, (
            f"noise-section hit for {query!r}: section={hit.sentence.section!r}, "
            f"sentence={hit.sentence.sentence[:100]!r}"
        )
        assert not hit.sentence.is_noise_section, (
            f"is_noise_section hit for {query!r}: {hit.sentence.section}"
        )
        assert not _is_reference_like(hit.sentence.sentence), (
            f"reference-list-like hit for {query!r}: "
            f"{hit.sentence.sentence[:100]!r}"
        )
        assert not is_retrieval_noise(
            hit.sentence.sentence, section=hit.sentence.section
        ), (
            f"title/reference-like hit for {query!r}: "
            f"section={hit.sentence.section!r}, "
            f"sentence={hit.sentence.sentence[:100]!r}"
        )

    if expected_paper is not None:
        keys = [h.sentence.citation_key for h in hits]
        matches = [k for k in keys if expected_paper.lower() in k.lower()]
        assert matches, (
            f"expected citation_key matching {expected_paper!r} in top-{k} "
            f"for {query!r}, got: {keys}"
        )


def test_no_duplicate_sentences_in_top10(idx: Index) -> None:
    """A single query should not return the same sentence text twice."""
    hits = idx.find("politeness varies across cultures", k=10)
    texts = [h.sentence.sentence for h in hits]
    assert len(texts) == len(set(texts)), (
        f"duplicate sentences in top-10: {texts}"
    )


def test_title_like_sentences_are_filtered() -> None:
    assert is_retrieval_noise(
        "Responding to compliments: A contrastive study of politeness strategies.",
        section="References",
    )
    assert is_retrieval_noise(
        "Modeling politeness variation across different social factors.",
        section="chunk_33",
    )
    assert not is_retrieval_noise(
        "This study investigated requests made by younger and older Chinese people on social media.",
        section="6. Conclusion",
    )


def test_find_can_filter_by_paper_and_section(idx: Index) -> None:
    hits = idx.find(
        "WeChat users prefer direct requests",
        k=5,
        paper="Liu",
        section="Conclusion",
        context_window=2,
    )
    assert hits
    for hit in hits:
        assert "liu" in (
            hit.sentence.paper_id
            + " "
            + hit.sentence.citation_key
            + " "
            + hit.sentence.pdf_path
        ).lower()
        assert "conclusion" in hit.sentence.section.lower()
    top = hits[0].to_dict()
    assert top["context_before"] or top["context_after"]


def test_find_can_hide_context(idx: Index) -> None:
    hits = idx.find(
        "politeness strategies vary across cultures",
        k=1,
        context_window=0,
    )
    assert hits
    top = hits[0].to_dict()
    assert top["context_before"] == ""
    assert top["context_after"] == ""


def test_verify_finds_exact_quote(idx: Index) -> None:
    """A known quote from the corpus should be findable verbatim."""
    hits = idx.verify(
        "preferred direct strategies when making requests on WeChat"
    )
    assert len(hits) >= 1
    assert "Liu" in hits[0].sentence.citation_key


def test_verify_returns_empty_for_nonsense(idx: Index) -> None:
    hits = idx.verify("this exact string xyzqqq should not exist")
    assert hits == []
