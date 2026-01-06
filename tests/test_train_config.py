from __future__ import annotations

import torch

from aslp.utils import coerce_float, load_yaml


def test_coerce_float_accepts_scientific_string() -> None:
    assert coerce_float("learning_rate", "2e-4") == 2e-4


def test_yaml_scientific_notation_safe_for_optimizer(tmp_path) -> None:
    cfg_path = tmp_path / "train.yaml"
    cfg_path.write_text("learning_rate: 2e-4\n", encoding="utf-8")
    cfg = load_yaml(str(cfg_path))
    lr = coerce_float("learning_rate", cfg["learning_rate"])
    assert isinstance(lr, float)
    param = torch.nn.Parameter(torch.zeros(1))
    torch.optim.AdamW([param], lr=lr)
