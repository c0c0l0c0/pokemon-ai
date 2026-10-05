"""
Writes training metrics to TensorBoard (checkpoints/<run>/tensorboard/) and log.jsonl.

View every run with:
    .venv/bin/tensorboard --logdir checkpoints
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.tensorboard import SummaryWriter


class RunLogger:
    def __init__(self, run_dir: Path, tensorboard: bool = True):
        self.jsonl_path = run_dir / "log.jsonl"
        self.writer = (
            SummaryWriter(log_dir=str(run_dir / "tensorboard")) if tensorboard else None
        )

    def scalars(self, step: int, values: dict[str, float]):
        """
        Tags like "loss/policy" are grouped by their prefix in TensorBoard. Called once
        per update, after the histograms and texts, so it also flushes everything to
        disk for TensorBoard to show it right away.
        """
        with open(self.jsonl_path, "a") as f:
            f.write(json.dumps({"update": step, **values}) + "\n")
        if self.writer is not None:
            for tag, value in values.items():
                self.writer.add_scalar(tag, value, step)
            self.writer.flush()

    def histogram(self, tag: str, values: np.ndarray, step: int, bins: Any = "auto"):
        if self.writer is not None and len(values):
            self.writer.add_histogram(tag, values, step, bins=bins)

    def text(self, tag: str, markdown: str, step: int):
        if self.writer is not None:
            self.writer.add_text(tag, markdown, step)

    def close(self):
        if self.writer is not None:
            self.writer.close()
