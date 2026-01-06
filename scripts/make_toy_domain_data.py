#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List


WORDS = [
    "apple",
    "river",
    "silver",
    "matrix",
    "orbit",
    "signal",
    "vector",
    "delta",
    "puzzle",
    "carbon",
]


def _make_example(rng: random.Random) -> Dict[str, str]:
    task_type = rng.choice(["reverse", "uppercase", "sum", "sort"])
    if task_type == "reverse":
        word = rng.choice(WORDS)
        instruction = "Reverse the word."
        input_text = word
        output = word[::-1]
    elif task_type == "uppercase":
        word = rng.choice(WORDS)
        instruction = "Uppercase the word."
        input_text = word
        output = word.upper()
    elif task_type == "sum":
        a = rng.randint(0, 50)
        b = rng.randint(0, 50)
        instruction = "Add the two numbers."
        input_text = f"a={a}, b={b}"
        output = str(a + b)
    else:
        nums = [rng.randint(0, 20) for _ in range(4)]
        instruction = "Sort the numbers ascending."
        input_text = ", ".join(str(n) for n in nums)
        output = ", ".join(str(n) for n in sorted(nums))

    text = (
        "### Instruction:\n"
        f"{instruction}\n"
        "### Input:\n"
        f"{input_text}\n"
        "### Response:\n"
        f"{output}"
    )
    return {"text": text}


def _write_jsonl(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate toy domain JSONL data")
    parser.add_argument("--out_dir", type=str, default="data")
    parser.add_argument("--n_train", type=int, default=200)
    parser.add_argument("--n_eval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    train_rows = [_make_example(rng) for _ in range(args.n_train)]
    eval_rows = [_make_example(rng) for _ in range(args.n_eval)]

    out_dir = Path(args.out_dir)
    _write_jsonl(out_dir / "domain_train.jsonl", train_rows)
    _write_jsonl(out_dir / "domain_eval.jsonl", eval_rows)


if __name__ == "__main__":
    main()
