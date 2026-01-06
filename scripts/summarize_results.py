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


def main() -> None:
    args = parse_args()
    rows: List[Dict[str, Any]] = []

    for run_dir in args.run_dir:
        run_path = Path(run_dir)
        general_path = run_path / args.general_name
        domain_path = run_path / args.domain_name
        if not general_path.exists() or not domain_path.exists():
            raise FileNotFoundError(f"Missing eval files in {run_dir}")
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
