from __future__ import annotations

import logging

from transformers import Trainer, TrainerCallback

from aslp.projector import ASLPProjector

logger = logging.getLogger(__name__)


class ProjectingCallback(TrainerCallback):
    def __init__(self, projector: ASLPProjector, proj_log_steps: int = 50) -> None:
        self.projector = projector
        self.proj_log_steps = proj_log_steps

    def on_optimizer_step(self, args, state, control, **kwargs):
        stats = self.projector.project()
        if stats:
            step = state.global_step + 1
            if self.proj_log_steps and step % self.proj_log_steps == 0:
                avg_ratio = sum(stats.values()) / len(stats)
                logger.info("ASLP projection ratio avg=%.6f", avg_ratio)
        return control


class ProjectingTrainer(Trainer):
    pass
