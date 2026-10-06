"""One-line progress display for long runs: bar, done of total, rate, elapsed and ETA, rewritten in place, plus a log line every
`log_every` seconds so the log file keeps the same progress when nobody watches the console."""
import logging
import shutil
import sys
import time
from datetime import timedelta


def clock(seconds: float) -> str:
    return str(timedelta(seconds=max(0, round(seconds))))


class Progress:
    def __init__(self, total: int, label: str, unit: str, log: logging.Logger, log_every: float = 30.0):
        self.total, self.label, self.unit, self.log, self.log_every = max(total, 1), label, unit, log, log_every
        self.done, self.start, self.last_log = 0, time.time(), 0.0
        self.show("")

    def step(self, n: int = 1, note: str = "") -> None:
        self.done += n
        self.show(note)

    def show(self, note: str) -> None:
        elapsed = time.time() - self.start
        rate = self.done / elapsed if elapsed > 0 and self.done else 0.0
        eta = (self.total - self.done) / rate if rate else 0.0
        filled = int(30 * self.done / self.total)
        line = (f"{self.label} [{'#' * filled}{'-' * (30 - filled)}] {100 * self.done / self.total:5.1f}% {self.done}/{self.total} {self.unit} "
                f"| {rate:.2f} {self.unit}/s | elapsed {clock(elapsed)} | ETA {clock(eta) if rate else '-'} | {note}")
        width = shutil.get_terminal_size((120, 20)).columns - 1
        sys.stdout.write("\r" + line[:width].ljust(width))
        sys.stdout.flush()
        if self.done and (time.time() - self.last_log >= self.log_every or self.done == self.total):
            self.log.info("%s: %d of %d %s, %.2f %s/s, elapsed %s, ETA %s %s", self.label, self.done, self.total, self.unit, rate, self.unit, clock(elapsed), clock(eta), note)
            self.last_log = time.time()

    def close(self) -> None:
        sys.stdout.write("\n")
        sys.stdout.flush()
        self.log.info("%s: finished %d %s in %s", self.label, self.done, self.unit, clock(time.time() - self.start))
