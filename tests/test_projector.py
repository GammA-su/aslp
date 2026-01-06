from __future__ import annotations

import torch

from aslp.projector import project_left, random_orthonormal


def test_projection_removes_component() -> None:
    torch.manual_seed(0)
    d = 32
    r_pca = 8
    r_lora = 4
    U = random_orthonormal(d, r_pca, seed=123)
    B = torch.randn(d, r_lora)
    B_proj = project_left(B, U)
    residual = torch.linalg.norm(U.T @ B_proj)
    assert residual.item() < 1e-5


def test_projection_idempotent() -> None:
    torch.manual_seed(1)
    d = 24
    r_pca = 6
    r_lora = 5
    U = random_orthonormal(d, r_pca, seed=7)
    B = torch.randn(d, r_lora)
    once = project_left(B, U)
    twice = project_left(once, U)
    assert torch.allclose(once, twice, atol=1e-6)
