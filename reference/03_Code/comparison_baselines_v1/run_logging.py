from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence


class CsvSink:
    def __init__(self, path: Path, fields: Sequence[str]):
        self.handle = path.open("x", encoding="utf-8-sig", newline="")
        self.writer = csv.DictWriter(self.handle, fieldnames=list(fields), extrasaction="ignore")
        self.writer.writeheader()
        self.count = 0

    def write(self, row: Mapping[str, Any]) -> None:
        self.writer.writerow(row)
        self.count += 1
        if self.count % 100 == 0:
            self.handle.flush()

    def close(self) -> None:
        self.handle.flush()
        self.handle.close()


class RunLogger:
    EPISODE_FIELDS = [
        "episode",
        "interaction_count",
        "reward",
        "success",
        "collision",
        "timeout",
        "truncated",
        "steps",
        "method",
        "seed",
    ]
    UPDATE_FIELDS = [
        "interaction_count",
        "update_count",
        "method",
        "loss",
        "policy_loss",
        "value_loss",
        "entropy",
        "mean_q",
        "mean_target",
        "mean_advantage",
        "mean_return",
        "gradient_norm",
        "learning_rate",
        "epsilon",
        "replay_size",
        "target_update_count",
        "rollout_interactions",
    ]

    def __init__(self, run_directory: Path):
        self.console = (run_directory / "run_log.txt").open("x", encoding="utf-8")
        self.episodes = CsvSink(run_directory / "training_episodes.csv", self.EPISODE_FIELDS)
        self.updates = CsvSink(run_directory / "training_updates.csv", self.UPDATE_FIELDS)

    def log(self, message: str) -> None:
        line = f"{datetime.now().isoformat(timespec='seconds')} {message}"
        print(line, flush=True)
        self.console.write(line + "\n")
        self.console.flush()

    def close(self) -> None:
        self.episodes.close()
        self.updates.close()
        self.console.flush()
        self.console.close()

