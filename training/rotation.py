import statistics


def _variance(scores):
    return statistics.pvariance(scores) if len(scores) > 1 else 0.0


def rank(results):

    return sorted(results, key=lambda w: _variance(results[w]))


def to_drop(results, k=30):
    return rank(results)[:k]
