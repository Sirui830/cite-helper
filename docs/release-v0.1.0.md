# cite-helper v0.1.0

Initial beta release of `cite-helper`: a local, sentence-level citation
retrieval CLI for academic PDF libraries.

## Highlights

- Build a searchable sentence index from a folder of PDFs.
- Search a draft claim and retrieve semantically similar source sentences.
- Verify whether a quote appears verbatim in the indexed corpus.
- Auto-build small PDF folders on first query.
- Build directly from a LitReview+ `source_index.jsonl` file.
- Store traceable sentence metadata including paper ID, citation key, section,
  source chunk ID, character offsets, and surrounding context.

## Commands

```bash
cite-helper build ./my-papers
cite-helper find "politeness strategies vary across cultures"
cite-helper verify "exact quote from a paper"
cite-helper stats
```

## Install

```bash
pip install git+https://github.com/Sirui830/cite-helper.git
```

For an isolated CLI install:

```bash
pipx install git+https://github.com/Sirui830/cite-helper.git
```

## Notes

- First run downloads `intfloat/multilingual-e5-small` and may take several
  minutes depending on network speed.
- PDF parsing is intentionally lightweight in v0.1. Complex layouts may leave
  line-break, hyphenation, or section-detection artifacts.
- Regression tests cover retrieval quality, noise filtering, duplicate
  handling, and quote verification against a local test index.
