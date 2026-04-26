from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Optional, Callable, TextIO


@dataclass
class _Step:
    name: str
    weight: float
    progressed: float = 0.0  # 0..100 within current step
    started_at: float = field(default_factory=time.monotonic)


class PipelineProgress:
    """
    Lightweight pipeline progress aggregator.

    - total_weight: sum of step weights mapped to 100% overall
    - on_change: callback receiving (overall_percent, current_label)

    Tracks elapsed time with a monotonic clock so it is robust to system
    clock changes, and exposes a best-effort ETA based on how much of the
    overall weight has been completed so far.
    """

    def __init__(
        self,
        total_weight: float = 100.0,
        on_change: Optional[Callable[[float, str], None]] = None,
    ) -> None:
        self.total_weight = float(total_weight)
        self.on_change = on_change
        self.completed_weight = 0.0
        self.current: Optional[_Step] = None
        self._started_at: float = time.monotonic()

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

    def elapsed_seconds(self) -> float:
        """Total seconds since the progress object was created."""
        return max(0.0, time.monotonic() - self._started_at)

    def eta_seconds(self) -> Optional[float]:
        """
        Best-effort remaining time estimate, or ``None`` when not yet
        computable (no progress recorded or already finished).
        """
        pct = self.overall_percent()
        if pct <= 0.0 or pct >= 100.0:
            return None
        elapsed = self.elapsed_seconds()
        if elapsed <= 0.0:
            return None
        total = elapsed * (100.0 / pct)
        return max(0.0, total - elapsed)

    def _emit(self) -> None:
        if self.on_change:
            self.on_change(self.overall_percent(), self.label())


def _format_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "--:--"
    seconds = int(max(0.0, seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def make_console_progress_printer(
    stream: Optional[TextIO] = None,
    progress: Optional[PipelineProgress] = None,
) -> Callable[[float, str], None]:
    """
    Build a console progress callback.

    When attached to a TTY the line redraws in place using a carriage return,
    and an ``elapsed | ETA`` suffix is appended whenever a ``PipelineProgress``
    instance is supplied. In non-TTY environments (logs, CI), each update is
    emitted on its own line for easy grepping.
    """
    out = stream if stream is not None else sys.stdout
    is_tty = bool(getattr(out, "isatty", lambda: False)()) and not os.environ.get("NO_COLOR_PROGRESS")

    def _printer(percent: float, label: str) -> None:
        suffix = ""
        if progress is not None:
            suffix = (
                f" | elapsed {_format_duration(progress.elapsed_seconds())}"
                f" | ETA {_format_duration(progress.eta_seconds())}"
            )
        line = f"[progress] {percent:6.2f}% | {label}{suffix}"
        if is_tty:
            out.write("\r" + line.ljust(80))
            if percent >= 100.0:
                out.write("\n")
            out.flush()
        else:
            print(line, file=out, flush=True)

    return _printer


