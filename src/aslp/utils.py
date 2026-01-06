from __future__ import annotations

import json
import logging
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

import numpy as np
import torch
import yaml


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return logging.getLogger("aslp")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_model_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def get_hf_token() -> Optional[str]:
    return os.environ.get("HF_TOKEN")


def load_yaml(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def coerce_float(name: str, value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError(f"Invalid value for {name}: {value!r}")
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid value for {name}: {value!r}") from None


def coerce_optional_float(name: str, value: Any) -> Optional[float]:
    if value is None:
        return None
    return coerce_float(name, value)


def save_json(path: str | Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def get_middle_third_indices(num_layers: int) -> Set[int]:
    start = num_layers // 3
    end = 2 * num_layers // 3
    return set(range(start, end))


def _parse_layer_index(name: str) -> Optional[int]:
    match = re.search(r"model\.layers\.(\d+)\.", name)
    if not match:
        return None
    return int(match.group(1))


def select_module_names(
    model: torch.nn.Module, module_suffixes: List[str], layer_indices: Optional[Set[int]]
) -> List[str]:
    names: List[str] = []
    for name, _module in model.named_modules():
        if not any(name.endswith(suffix) for suffix in module_suffixes):
            continue
        if layer_indices is not None:
            layer_idx = _parse_layer_index(name)
            if layer_idx is None or layer_idx not in layer_indices:
                continue
        names.append(name)
    return names
