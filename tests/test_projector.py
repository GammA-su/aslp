from __future__ import annotations

import torch

from aslp.projector import ASLPProjector, project_left, random_orthonormal


class _FakeLoraModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lora_B = torch.nn.ModuleDict({"default": torch.nn.Linear(2, 2, bias=False)})


class _FakeAttn(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.o_proj = _FakeLoraModule()


class _FakeLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _FakeAttn()


class _FakeInnerModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([_FakeLayer()])


class _FakeWrapper(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _FakeInnerModel()


class _FakeOuterModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base_model = _FakeWrapper()


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


def test_projector_matches_prefixed_module_names() -> None:
    model = _FakeOuterModel()
    u_by_module = {"model.layers.0.self_attn.o_proj": torch.eye(2)}
    projector = ASLPProjector(model, u_by_module)
    ratios = projector.project()
    assert "model.layers.0.self_attn.o_proj" in ratios
