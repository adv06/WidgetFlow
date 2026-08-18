import math

from credit.credit import WEIGHTS, BANDS


class Controller:


    def __init__(self, base=None, *, lr=0.3, temp=1.0, floor=0.02):
        base = base or {b: WEIGHTS[b] for b in BANDS}
        s = sum(base.values())
        self.base = {b: v / s for b, v in base.items()}
        self.w = dict(self.base)
        self.lr, self.temp, self.floor = lr, temp, floor

    def update(self, band_means):
        m, keys = band_means, list(self.w)
        vals = [m[b] for b in keys]
        mu = sum(vals) / len(vals)
        sd = (sum((v - mu) ** 2 for v in vals) / len(vals)) ** 0.5 or 1.0

        tgt = {b: self.base[b] * math.exp(-((m[b] - mu) / sd) / self.temp) for b in keys}
        z = sum(tgt.values())
        tgt = {b: t / z for b, t in tgt.items()}
        self.w = {b: (1 - self.lr) * self.w[b] + self.lr * tgt[b] for b in keys}
        self.w = {b: max(self.floor, v) for b, v in self.w.items()}
        z = sum(self.w.values())
        self.w = {b: v / z for b, v in self.w.items()}
        return dict(self.w)
