#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize evaluation results")
    parser.add_argument("--run_dir", action="append", required=True)
    parser.add_argument("--general_name", type=str, default="general.json")
    parser.add_argument("--domain_name", type=str, default="domain.json")
    return parser.parse_args()


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8").strip())


def _format_table(rows: List[Dict[str, Any]]) -> str:
    headers = ["mode", "seed", "general_ppl", "domain_ppl"]
    cols = {h: [h] for h in headers}
    for row in rows:
        cols["mode"].append(str(row["mode"]))
        cols["seed"].append(str(row["seed"]))
        cols["general_ppl"].append(f"{row['general_ppl']:.4f}")
        cols["domain_ppl"].append(f"{row['domain_ppl']:.4f}")
    widths = {h: max(len(v) for v in cols[h]) for h in headers}
    lines = []
    header_line = "  ".join(h.ljust(widths[h]) for h in headers)
    lines.append(header_line)
    lines.append("  ".join("-" * widths[h] for h in headers))
    for row in rows:
        line = "  ".join(
            [
                str(row["mode"]).ljust(widths["mode"]),
                str(row["seed"]).ljust(widths["seed"]),
                f"{row['general_ppl']:.4f}".ljust(widths["general_ppl"]),
                f"{row['domain_ppl']:.4f}".ljust(widths["domain_ppl"]),
            ]
        )
        lines.append(line)
    return "\n".join(lines)


def _find_run_dirs(run_path: Path, general_name: str, domain_name: str) -> List[Path]:
    direct_general = run_path / general_name
    direct_domain = run_path / domain_name
    if direct_general.exists() and direct_domain.exists():
        return [run_path]

    if run_path.is_dir():
        candidates = []
        for child in sorted(run_path.iterdir(), key=lambda p: p.name):
            if not child.is_dir():
                continue
            child_general = child / general_name
            child_domain = child / domain_name
            if child_general.exists() and child_domain.exists():
                candidates.append(child)
        if candidates:
            return candidates

    raise FileNotFoundError(f"Missing eval files in {run_path}")


def main() -> None:
    args = parse_args()
    rows: List[Dict[str, Any]] = []

    for run_dir in args.run_dir:
        run_path = Path(run_dir)
        for resolved_run in _find_run_dirs(run_path, args.general_name, args.domain_name):
            general_path = resolved_run / args.general_name
            domain_path = resolved_run / args.domain_name
            general = _load_json(general_path)
            domain = _load_json(domain_path)
            row = {
                "mode": general.get("mode", domain.get("mode", "unknown")),
                "seed": general.get("seed", domain.get("seed", -1)),
                "general_ppl": general["ppl"],
                "domain_ppl": domain["ppl"],
            }
            rows.append(row)
            print(json.dumps(row))

    print(_format_table(rows))


if __name__ == "__main__":
    main()
