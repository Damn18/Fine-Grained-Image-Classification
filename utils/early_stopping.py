"""
Early stopping monitor.

Halts training when the tracked metric (validation loss by default) has not
improved by more than `min_delta` for `patience` consecutive epochs.
"""


class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 0.0, mode: str = "min"):
        if mode not in ("min", "max"):
            raise ValueError("mode must be 'min' or 'max'")
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best: float | None = None
        self.counter: int = 0
        self.triggered: bool = False

    def __call__(self, value: float) -> bool:
        """Return True if training should stop."""
        if self.best is None:
            self.best = value
            return False

        improved = (
            (value < self.best - self.min_delta)
            if self.mode == "min"
            else (value > self.best + self.min_delta)
        )

        if improved:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.triggered = True
                return True

        return False
