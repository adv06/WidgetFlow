import random
class TrainFilter:
    def __init__(self, pool, size=130, seed=None):
        self.rng = random.Random(seed)
        self.pool = list(pool)
        self.size = size
        self.seen = set()
        self.training_set = self._draw(size)

    def _draw(self, n):
        unseen = [w for w in self.pool if w not in self.seen]
        n = min(n, len(unseen))
        picked = self.rng.sample(unseen, n)
        self.seen.update(picked)
        return picked

    def swap(self, drop):
        drop = set(drop)
        self.training_set = [w for w in self.training_set if w not in drop]
        self.training_set += self._draw(len(drop))
        return self.training_set
