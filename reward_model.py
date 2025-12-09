import re

class MathRewardModel:
    """Simple rule-based reward model for arithmetic problems.

    Each sample is a dict with:
      - prompt: the question (e.g. "What is 2 + 3?")
      - answer: the correct numeric answer as string (e.g. "5")

    The completion is the model output. We try to extract the final integer
    from the completion and compare to the ground truth.
    """

    def __init__(self, correct_reward: float = 1.0, wrong_reward: float = 0.0):
        self.correct_reward = correct_reward
        self.wrong_reward = wrong_reward

    def _extract_int(self, text: str):
        # Find the last integer in the string, if any
        matches = re.findall(r"-?\d+", text)
        if not matches:
            return None
        try:
            return int(matches[-1])
        except ValueError:
            return None

    def compute_reward(self, sample, completion: str) -> float:
        """Return a scalar reward for a (sample, completion) pair."""
        target = self._extract_int(sample.get("answer", ""))
        pred = self._extract_int(completion)

        if target is None or pred is None:
            return self.wrong_reward
        return self.correct_reward if target == pred else self.wrong_reward
