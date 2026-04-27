"""Simple AI opponent: randomly punches, kicks, or idles.

Behavior:
  - Mostly idle (auto-blocks incoming attacks)
  - Randomly throws a punch or kick every 1-3 seconds
  - After attacking, returns to idle for a cooldown period
"""

import random
import time


def make_ai_provider(attack_interval=(1.0, 3.0), attack_duration=0.15):
    next_attack_at = time.monotonic() + random.uniform(*attack_interval)
    state = {"action": "idle", "until": 0.0}

    def provider():
        now = time.monotonic()
        nonlocal next_attack_at

        if now < state["until"]:
            return state["action"]

        if now >= next_attack_at:
            state["action"] = random.choice(["lpunch", "rpunch", "forward", "backward"])
            state["until"] = now + attack_duration
            next_attack_at = now + attack_duration + random.uniform(*attack_interval)
            return state["action"]

        state["action"] = "idle"
        return "idle"

    return provider
