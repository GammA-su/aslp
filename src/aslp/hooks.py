from __future__ import annotations

import logging
from typing import Any, Callable, List, Optional

import torch
from torch.utils.hooks import RemovableHandle

logger = logging.getLogger(__name__)


def extract_tensor(output: Any) -> Optional[torch.Tensor]:
    if isinstance(output, (tuple, list)):
        output = output[0] if output else None
    if output is None or not isinstance(output, torch.Tensor):
        return None
    return output


def _make_hook(name: str, callback: Callable[[str, Any], None]) -> Callable[..., None]:
    def hook(_module: torch.nn.Module, _inputs: Any, output: Any) -> None:
        callback(name, output)

    return hook


def register_activation_hooks(
    model: torch.nn.Module, module_names: List[str], callback: Callable[[str, Any], None]
) -> List[RemovableHandle]:
    handles: List[RemovableHandle] = []
    target_set = set(module_names)
    found = set()
    for name, module in model.named_modules():
        if name in target_set:
            handles.append(module.register_forward_hook(_make_hook(name, callback)))
            found.add(name)
    missing = target_set - found
    if missing:
        logger.warning("Missing %d target modules for hooks", len(missing))
    return handles
