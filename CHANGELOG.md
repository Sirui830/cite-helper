# Changelog

## 0.1.0 (2026-06-05)

Initial release.

- PDF folder indexing via `pymupdf4llm` + `pysbd` + `multilingual-e5-small`
- CLI commands: `build`, `find`, `verify`, `stats`
- `find --auto-build` (default on, with safety threshold for large folders)
- Build from LitReview+ `source_index.jsonl` as alternative input
- 24-test regression suite covering retrieval quality + dedup + verify
- opencode / Claude Code / Cursor integration docs
