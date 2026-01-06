from __future__ import annotations

import logging
from typing import Optional

from transformers import Trainer

from aslp.projector import ASLPProjector

logger = logging.getLogger(__name__)


class ProjectingTrainer(Trainer):
    def __init__(self, *args, projector: Optional[ASLPProjector] = None, proj_log_steps: int = 50, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.projector = projector
        self.proj_log_steps = proj_log_steps

    def optimizer_step(self, *args, **kwargs):
        output = super().optimizer_step(*args, **kwargs)
        if self.projector is not None:
            stats = self.projector.project()
            if stats and self.state.global_step % self.proj_log_steps == 0:
                avg_ratio = sum(stats.values()) / len(stats)
                logger.info("ASLP projection ratio avg=%.6f", avg_ratio)
        return output
