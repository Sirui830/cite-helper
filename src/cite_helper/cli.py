"""cite-helper command-line entrypoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _print_hit_human(hit_dict: dict) -> None:
    rank = hit_dict["rank"]
    score = hit_dict["score"]
    paper = hit_dict["paper_id"]
    citation = hit_dict["citation_key"]
    section = hit_dict["section"]
    sentence = hit_dict["sentence"]
    ctx_before = hit_dict.get("context_before", "")
    ctx_after = hit_dict.get("context_after", "")
    sent_id = hit_dict["sent_id"]
    print(f"\n[{rank}] score={score:.3f}  [{citation}] · {section}")
    print(f"    {sentence}")
    if ctx_before or ctx_after:
        print(f"    context: ...{ctx_before} >>>{sentence}<<< {ctx_after}...")
    print(f"    sent_id={sent_id}  paper={paper}")


def cmd_build(args: argparse.Namespace) -> int:
    from .indexer import build_from_pdf_folder, build_from_source_index

    if args.from_source_index:
        report = build_from_source_index(
            source_index_path=Path(args.from_source_index),
            output_folder=Path(args.target),
        )
    else:
        folder = Path(args.target)
        if not folder.is_dir():
            print(f"Error: {folder} is not a directory", file=sys.stderr)
            return 1
        report = build_from_pdf_folder(folder=folder)

    print(f"\n=== Index built ===")
    for k, v in report.items():
        print(f"  {k}: {v}")
    return 0


def cmd_find(args: argparse.Namespace) -> int:
    from .indexer import build_from_pdf_folder, INDEX_DIRNAME
    from .retriever import Index

    folder = Path(args.folder).resolve()
    idx_dir = folder / INDEX_DIRNAME

    if not idx_dir.exists():
        if args.no_auto_build:
            print(
                f"Error: no index at {idx_dir}. Run `cite-helper build {folder}` "
                "first, or omit --no-auto-build.",
                file=sys.stderr,
            )
            return 1

        pdfs = list(folder.glob("*.pdf"))
        n_pdfs = len(pdfs)
        if n_pdfs == 0:
            print(
                f"Error: no PDFs in {folder} and no index. Nothing to do.",
                file=sys.stderr,
            )
            return 1

        if n_pdfs > args.auto_build_max and not args.yes:
            print(
                f"⚠️  Folder has {n_pdfs} PDFs — auto-building would take "
                f"approximately {max(1, n_pdfs // 5)}-{max(2, n_pdfs // 3)} "
                f"minutes.\n"
                f"   Re-run with --yes to confirm, OR run "
                f"`cite-helper build {folder}` separately, OR pass "
                f"--auto-build-max {n_pdfs} to raise the threshold.",
                file=sys.stderr,
            )
            return 2

        print(
            f"No index in {folder}. Auto-building ({n_pdfs} PDFs)...",
            file=sys.stderr,
        )
        try:
            build_from_pdf_folder(folder=folder)
        except Exception as exc:
            print(f"Auto-build failed: {exc}", file=sys.stderr)
            return 1
        print("Auto-build done. Running query...\n", file=sys.stderr)

    try:
        idx = Index(folder)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    context_window = 0 if args.no_context else max(0, args.show_context)
    hits = idx.find(
        args.query,
        k=args.k,
        paper=args.paper,
        section=args.section,
        context_window=context_window,
        include_noise=args.include_noise,
    )
    hit_dicts = [h.to_dict() for h in hits]

    if args.json:
        print(json.dumps(hit_dicts, ensure_ascii=False, indent=2))
        return 0

    if not hits:
        print("No matches.")
        return 1

    print(f"Top {len(hits)} matches for: {args.query!r}")
    for h in hit_dicts:
        _print_hit_human(h)
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    from .retriever import Index

    try:
        idx = Index(Path(args.folder))
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    hits = idx.verify(args.text)
    hit_dicts = [h.to_dict() for h in hits]

    if args.json:
        print(json.dumps(hit_dicts, ensure_ascii=False, indent=2))
        return 0 if hits else 1

    if not hits:
        print(f"Not found in corpus: {args.text!r}")
        return 1

    print(f"Found {len(hits)} occurrence(s) of: {args.text!r}")
    for h in hit_dicts:
        _print_hit_human(h)
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    from .retriever import Index

    try:
        idx = Index(Path(args.folder))
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(idx.meta, ensure_ascii=False, indent=2))
    print(f"\nLoaded {len(idx.sentences)} sentences, "
          f"embeddings shape: {idx.embeddings.shape}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="cite-helper",
        description="Sentence-level citation retrieval over a folder of PDFs.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build", help="Build the search index")
    p_build.add_argument(
        "target",
        help="PDF folder, OR target folder for --from-source-index mode",
    )
    p_build.add_argument(
        "--from-source-index",
        metavar="PATH",
        default=None,
        help="Skip PDF parse; build from a LitReview+ source_index.jsonl instead",
    )
    p_build.set_defaults(func=cmd_build)

    p_find = sub.add_parser("find", help="Search for citations matching a query")
    p_find.add_argument("query", help="Your sentence / claim text")
    p_find.add_argument(
        "--folder", "-f",
        default=".",
        help="Folder containing .cite_helper_index (default: current dir)",
    )
    p_find.add_argument(
        "-k", "--top",
        type=int,
        default=5,
        help="Number of results (default: 5)",
        dest="k",
    )
    p_find.add_argument("--json", action="store_true", help="Output as JSON")
    p_find.add_argument(
        "--paper",
        default=None,
        help=(
            "Only search papers whose paper_id, citation_key, or PDF filename "
            "contains this text"
        ),
    )
    p_find.add_argument(
        "--section",
        default=None,
        help="Only search sections whose label contains this text",
    )
    p_find.add_argument(
        "--show-context",
        type=int,
        default=1,
        metavar="N",
        help="Show N neighboring indexed sentences on each side (default: 1)",
    )
    p_find.add_argument(
        "--no-context",
        action="store_true",
        help="Hide surrounding context in human-readable output",
    )
    p_find.add_argument(
        "--include-noise",
        action="store_true",
        help="Include title/reference-like sentences normally hidden from results",
    )
    p_find.add_argument(
        "--no-auto-build",
        action="store_true",
        help="Refuse to build index if missing (default: auto-build if folder has few PDFs)",
    )
    p_find.add_argument(
        "--auto-build-max",
        type=int,
        default=30,
        help="Max PDFs to auto-build without confirmation (default: 30)",
    )
    p_find.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Confirm auto-build even when PDF count exceeds --auto-build-max",
    )
    p_find.set_defaults(func=cmd_find)

    p_verify = sub.add_parser(
        "verify",
        help="Confirm a quote verbatim exists in the corpus",
    )
    p_verify.add_argument("text", help="Exact quote text to search for")
    p_verify.add_argument(
        "--folder", "-f",
        default=".",
        help="Folder containing .cite_helper_index (default: current dir)",
    )
    p_verify.add_argument("--json", action="store_true", help="Output as JSON")
    p_verify.set_defaults(func=cmd_verify)

    p_stats = sub.add_parser("stats", help="Show index metadata")
    p_stats.add_argument(
        "--folder", "-f",
        default=".",
        help="Folder containing .cite_helper_index (default: current dir)",
    )
    p_stats.set_defaults(func=cmd_stats)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
