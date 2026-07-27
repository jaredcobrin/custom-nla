import random

# Visible-fraction ladder: 1.0 = full prefix shown, 0.0 = nothing shown.
LEVELS = [1.0, 0.75, 0.5, 0.25, 0.0]


class VisibleFractionCurriculum:
    """Tracks the current visible-fraction level and gates advancement on a
    rolling window of held-out, zero-context FVE probes (see curriculum_plan.md
    - in-curriculum FVE can be inflated by copying/paraphrasing visible text,
    so advancement must be judged on a probe where there's nothing to copy)."""

    def __init__(self, levels=None, advance_threshold=0.6, window=20):
        self.levels = levels if levels is not None else list(LEVELS)
        self.level_index = 0
        self.advance_threshold = advance_threshold
        self.window = window
        self.fve_history = []
        self.probes_at_level = 0

    @property
    def fraction(self):
        return self.levels[self.level_index]

    def sample_context(self, text):
        """Prefix of `text` up to a randomized cutoff under the current
        fraction's ceiling - randomized so AV can't learn a shortcut tied to
        one specific truncation amount."""
        fraction = self.fraction
        if fraction <= 0.0 or not text:
            return ""
        actual_fraction = random.uniform(fraction * 0.75, fraction)
        cutoff = int(actual_fraction * len(text))
        return text[:cutoff]

    def record_probe_fve(self, fve):
        """Feed in one zero-context probe FVE reading. Returns True if this
        reading triggered an advance to the next (harder) level.

        Hold time before advancing is `window` probes (= window * EVAL_INTERVAL
        training steps), not a separate step-count parameter - fve_history is
        reset to empty on every advance, so it has to refill to `window` again
        before another advance can even be considered. That's the hysteresis."""
        self.probes_at_level += 1
        self.fve_history.append(fve)
        if len(self.fve_history) > self.window:
            self.fve_history.pop(0)

        at_max_level = self.level_index >= len(self.levels) - 1
        ready = len(self.fve_history) >= self.window
        if not at_max_level and ready:
            rolling_mean = sum(self.fve_history) / len(self.fve_history)
            if rolling_mean >= self.advance_threshold:
                self.level_index += 1
                self.probes_at_level = 0
                self.fve_history = []
                return True
        return False

    def state_dict(self):
        return {
            "level_index": self.level_index,
            "probes_at_level": self.probes_at_level,
            "fve_history": self.fve_history,
        }

    def load_state_dict(self, state):
        self.level_index = state["level_index"]
        self.probes_at_level = state["probes_at_level"]
        self.fve_history = state["fve_history"]
