from __future__ import annotations

import torch
from transformers import TrainerControl, TrainerState, TrainingArguments

from aslp.projector import ASLPProjector
from aslp.trainer_patch import ProjectingCallback


class _ToyLoraModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lora_B = torch.nn.ModuleDict({"default": torch.nn.Linear(2, 2, bias=False)})


class _ToyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block = _ToyLoraModule()


def test_projecting_callback_updates_lora_B(tmp_path) -> None:
    model = _ToyModel()
    param = model.block.lora_B["default"].weight
    with torch.no_grad():
        param.copy_(torch.tensor([[1.0, -1.0], [0.5, -0.5]]))

    projector = ASLPProjector(model, {"block": torch.eye(2)})
    callback = ProjectingCallback(projector, proj_log_steps=1)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss = param.sum()
    loss.backward()
    optimizer.step()

    after_step = param.detach().clone()
    args = TrainingArguments(output_dir=str(tmp_path))
    state = TrainerState()
    control = TrainerControl()
    callback.on_optimizer_step(args, state, control)
    after_project = param.detach().clone()

    assert not torch.allclose(after_step, after_project)
    assert torch.allclose(after_project, torch.zeros_like(after_project), atol=1e-6)
