from __future__ import annotations

from typing import Dict

import torch


def random_orthonormal(d: int, r: int, seed: int) -> torch.Tensor:
    if r <= 0 or d <= 0:
        raise ValueError("d and r must be positive")
    if r > d:
        raise ValueError("r must be <= d")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    mat = torch.randn((d, r), generator=gen, dtype=torch.float32)
    q, _ = torch.linalg.qr(mat)
    return q[:, :r].contiguous()


def project_left(B: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
    return B - U @ (U.T @ B)


def projection_ratio(B: torch.Tensor, U: torch.Tensor) -> float:
    if B.numel() == 0:
        return 0.0
    utb = U.T @ B
    denom = torch.linalg.norm(B) + 1e-8
    return float(torch.linalg.norm(utb) / denom)


class ASLPProjector:
    def __init__(self, model: torch.nn.Module, u_by_module: Dict[str, torch.Tensor], adapter_name: str = "default") -> None:
        self.model = model
        self.u_by_module = u_by_module
        self.adapter_name = adapter_name
        self.module_to_B: Dict[str, torch.nn.Parameter] = {}

        lora_B_by_name: Dict[str, torch.nn.Parameter] = {}
        for name, module in model.named_modules():
            if hasattr(module, "lora_B") and adapter_name in module.lora_B:
                lora_B_by_name[name] = module.lora_B[adapter_name].weight

        missing = []
        ambiguous = []
        for name in u_by_module.keys():
            if name in lora_B_by_name:
                self.module_to_B[name] = lora_B_by_name[name]
                continue
            matches = [full_name for full_name in lora_B_by_name.keys() if full_name.endswith(name)]
            if len(matches) == 1:
                self.module_to_B[name] = lora_B_by_name[matches[0]]
            elif len(matches) > 1:
                ambiguous.append(name)
            else:
                missing.append(name)
        if ambiguous:
            raise ValueError(f"Ambiguous LoRA B weights for modules: {ambiguous}")
        if missing:
            raise ValueError(f"Missing LoRA B weights for modules: {missing}")

    def project(self) -> Dict[str, float]:
        ratios: Dict[str, float] = {}
        with torch.no_grad():
            for name, B in self.module_to_B.items():
                U = self.u_by_module.get(name)
                if U is None:
                    continue
                U_dev = U.to(device=B.device, dtype=B.dtype)
                B.data.copy_(project_left(B.data, U_dev))
                ratios[name] = projection_ratio(B.data, U_dev)
        return ratios
