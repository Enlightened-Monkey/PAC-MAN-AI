"""Learning-rate schedules for PPO / MaskablePPO."""


def constant_schedule(value: float):
    """Fixed LR for the whole training run (avoids decay-to-zero on resume segments)."""

    def _f(_progress_remaining: float) -> float:
        return value

    return _f


def linear_schedule(initial: float):
    """Linear LR decay from ``initial`` to 0 over each ``learn()`` call."""

    def _f(progress_remaining: float) -> float:
        return initial * progress_remaining

    return _f
