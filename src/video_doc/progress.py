from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Callable


@dataclass
class _Step:
    name: str
    weight: float
    progressed: float = 0.0  # 0..100 within current step


class PipelineProgress:
    """
    Lightweight pipeline progress aggregator.

    - total_weight: sum of step weights mapped to 100% overall
    - on_change: callback receiving (overall_percent, current_label)
    """

    def __init__(self, total_weight: float = 100.0, on_change: Optional[Callable[[float, str], None]] = None) -> None:
        self.total_weight = float(total_weight)
        self.on_change = on_change
        self.completed_weight = 0.0
        self.current: Optional[_Step] = None

    def start_step(self, name: str, weight: float) -> None:
        if self.current is not None:
            self.end_step()
        self.current = _Step(name=name, weight=float(weight), progressed=0.0)
        self._emit()

    def update(self, percent_within_step: float) -> None:
        if self.current is None:
            return
        p = max(0.0, min(100.0, float(percent_within_step)))
        if p <= self.current.progressed:
            return
        self.current.progressed = p
        self._emit()

    def end_step(self) -> None:
        if self.current is None:
            return
        self.completed_weight += self.current.weight
        self.current = None
        self._emit()

    def overall_percent(self) -> float:
        if self.total_weight <= 0:
            return 0.0
        in_progress = 0.0
        if self.current is not None and self.current.weight > 0:
            in_progress = (self.current.progressed / 100.0) * self.current.weight
        return max(0.0, min(100.0, 100.0 * (self.completed_weight + in_progress) / self.total_weight))

    def label(self) -> str:
        if self.current is None:
            return "Idle"
        return self.current.name

    def _emit(self) -> None:
        if self.on_change:
            self.on_change(self.overall_percent(), self.label())


def make_console_progress_printer() -> Callable[[float, str], None]:
    def _printer(percent: float, label: str) -> None:
        print(f"Progress: {percent:5.1f}% - {label}", flush=True)
    return _printer


