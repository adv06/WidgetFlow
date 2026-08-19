import math

from credit.credit import WEIGHTS, BANDS


class Controller:
    def __init__(self, base=None, *, lr=0.3, temp=1.0, floor=0.02,
                 w_rel=1.0, w_delta=0.6, w_goal=0.5, goal=100.0):
        base = base or {b: WEIGHTS[b] for b in BANDS}
        s = sum(base.values())
        self.base = {b: v / s for b, v in base.items()}
        self.w = dict(self.base)
        self.prev = None
        self.lr, self.temp, self.floor = lr, temp, floor
        self.w_rel, self.w_delta, self.w_goal, self.goal = w_rel, w_delta, w_goal, goal

    @staticmethod
    def _standardize(d, keys):
        vals = [d[b] for b in keys]
        mu = sum(vals) / len(vals)
        sd = (sum((v - mu) ** 2 for v in vals) / len(vals)) ** 0.5 or 1.0
        return {b: (d[b] - mu) / sd for b in keys}

    def update(self, band_means):
        m, keys = band_means, list(self.w)

        rel = {b: -v for b, v in self._standardize(m, keys).items()}

        if self.prev is not None:
            worse = {b: (self.prev[b] - m[b]) for b in keys}
            dlt = self._standardize(worse, keys)
        else:
            dlt = {b: 0.0 for b in keys}

        gap = {b: max(0.0, (self.goal - m[b]) / self.goal) for b in keys}
        gap = self._standardize(gap, keys)

        need = {b: self.w_rel * rel[b] + self.w_delta * dlt[b] + self.w_goal * gap[b]
                for b in keys}

        tgt = {b: self.base[b] * math.exp(need[b] / self.temp) for b in keys}
        z = sum(tgt.values())
        tgt = {b: t / z for b, t in tgt.items()}

        self.w = {b: (1 - self.lr) * self.w[b] + self.lr * tgt[b] for b in keys}
        self.w = {b: max(self.floor, v) for b, v in self.w.items()}
        z = sum(self.w.values())
        self.w = {b: v / z for b, v in self.w.items()}

        self.prev = dict(m)
        return dict(self.w)