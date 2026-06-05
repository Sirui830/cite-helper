# Changelog

## Unreleased

- Add focused retrieval filters: `find --paper` and `find --section`
- Add context display controls: `find --show-context N` and `find --no-context`
- Hide title/reference-like sentences from default search results, with
  `--include-noise` as an escape hatch
- Expand regression coverage for focused filters and title-like noise

## 0.1.0 (2026-06-05)

Initial release.

- PDF folder indexing via `pymupdf4llm` + `pysbd` + `multilingual-e5-small`
- CLI commands: `build`, `find`, `verify`, `stats`
- `find --auto-build` (default on, with safety threshold for large folders)
- Build from LitReview+ `source_index.jsonl` as alternative input
- 24-test regression suite covering retrieval quality + dedup + verify
- opencode / Claude Code / Cursor integration docs
