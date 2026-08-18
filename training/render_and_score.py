from dataclasses import dataclass
from statistics import mean

from credit.credit import BandScorer, WEIGHTS


@dataclass
class Scored:
    rollout: object
    bands: dict
    reward: float
    below_mean: bool = False


def render_and_score(widget, rollouts, weights=None, *, score_fn=None):

    weights = weights or WEIGHTS
    scored = []
    with BandScorer(widget, scorer=score_fn) as s:
        for r in rollouts:
            code = r.text if hasattr(r, "text") else r
            bands = s(code) or {}
            reward = sum(weights.get(b, 0.0) * v for b, v in bands.items())
            scored.append(Scored(r, bands, reward))
    if scored:
        m = mean(x.reward for x in scored)
        for x in scored:
            x.below_mean = x.reward < m
    return scored
