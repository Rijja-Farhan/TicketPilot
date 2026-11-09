"""python -m ticketpilot.ingest [--kb DIR] [--out DIR] [--report FILE]"""
import argparse
from pathlib import Path

from ticketpilot.ingest.pipeline import (DEFAULT_INDEX, DEFAULT_KB, DEFAULT_REPORT, build_indexes, ingest, score_pairs,
                                      write_report)


def main() -> None:
    p = argparse.ArgumentParser(prog="ticketpilot.ingest")
    p.add_argument("--kb", type=Path, default=DEFAULT_KB)
    p.add_argument("--out", type=Path, default=DEFAULT_INDEX)
    p.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = p.parse_args()
    result = ingest(args.kb)
    counts = build_indexes(result.docs, args.out)
    write_report(result, counts, args.report, score_pairs())
    print(f"{result.files_in} files -> {len(result.docs)} docs, {len(result.dropped)} duplicates dropped, "
          f"{len(result.quarantined)} quarantined; chunks {counts}; report {args.report}")


if __name__ == "__main__":
    main()
