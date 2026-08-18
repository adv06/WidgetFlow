from dataclasses import dataclass
from statistics import mean

from credit.credit import BandScorer, WEIGHTS


@dataclass
class Scored:
    rollout: object              # the Policy.Rollout that produced this
    bands: dict                  # {band: score vs target}; {} if the render failed
    reward: float                # dynamic weighted reward: sum_b w_b * band_b
    below_mean: bool = False     # negative advantage -> goes to Credit Weighting


def render_and_score(widget, rollouts, weights=None, *, score_fn=None):
    """Render each rollout's JSX (Playwright, via BandScorer) against the widget's
    target and score it on the dynamic band weights. Returns [Scored], with the
    below-group-mean rollouts flagged (those feed Credit Weighting)."""
    weights = weights or WEIGHTS
    scored = []
    with BandScorer(widget, scorer=score_fn) as s:      # one browser for all N rollouts
        for r in rollouts:
            code = r.text if hasattr(r, "text") else r
            bands = s(code) or {}                        # None (broken render) -> {} -> reward 0
            reward = sum(weights.get(b, 0.0) * v for b, v in bands.items())
            scored.append(Scored(r, bands, reward))
    if scored:
        m = mean(x.reward for x in scored)
        for x in scored:
            x.below_mean = x.reward < m
    return scored
