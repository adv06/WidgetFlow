import statistics


def _variance(scores):
    return statistics.pvariance(scores) if len(scores) > 1 else 0.0


def rank(results):
    # results: {widget_id: [band-weighted score, ...]} -> ids, lowest variance first
    return sorted(results, key=lambda w: _variance(results[w]))


def to_drop(results, k=30):
    return rank(results)[:k]
