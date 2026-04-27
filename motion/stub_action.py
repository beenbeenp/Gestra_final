"""Deterministic action provider for end-to-end smoke testing without a webcam.

Cycles through the action vocabulary on a fixed cadence so we can verify that
each named action is correctly translated by Street-Pyter's external-action
seam without needing a trained model or live pose data. Activated by setting
GESTRA_STUB_ACTION=1 when launching main.py.
"""

import time

# Schedule: (action_name, hold_seconds). idle is the resting state.
DEFAULT_SCHEDULE = [
    ("idle", 1.5),
    ("forward", 0.6),
    ("idle", 0.8),
    ("backward", 0.6),
    ("idle", 0.8),
    ("punch", 0.3),
    ("idle", 1.0),
    ("kick", 0.3),
    ("idle", 1.5),
]


def make_stub_provider(schedule=None):
    schedule = list(schedule or DEFAULT_SCHEDULE)
    total = sum(d for _, d in schedule)
    start = time.monotonic()

    def provider():
        elapsed = (time.monotonic() - start) % total
        cursor = 0.0
        for name, dur in schedule:
            cursor += dur
            if elapsed < cursor:
                return name
        return "idle"

    return provider
