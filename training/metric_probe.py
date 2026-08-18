import torch
from statistics import mean

from credit.credit import BandScorer, BANDS


@torch.no_grad()
def _greedy(policy, widget):
    enc = policy._inputs(widget)
    plen = enc["input_ids"].shape[1]
    out = policy.model.generate(**enc, do_sample=False, max_new_tokens=policy.max_new_tokens)
    return policy.processor.decode(out[0, plen:], skip_special_tokens=True)


@torch.no_grad()
def metric_probe(policy, widgets, *, baseline=None, delta=0.0, score_fn=None):
    was_training = policy.model.training
    policy.model.eval()
    totals = {b: [] for b in BANDS}
    try:
        for image, gt_path in widgets:          # each val widget: (PIL image, target png)
            with BandScorer(gt_path, scorer=score_fn) as s:
                bands = s(_greedy(policy, image)) or {}
            for b in BANDS:
                totals[b].append(bands.get(b, 0.0))     # failed / missing band -> 0
    finally:
        policy.model.train(was_training)

    means = {b: (mean(v) if v else 0.0) for b, v in totals.items()}
    passed = None
    if baseline is not None:
        passed = all(means[b] >= baseline.get(b, 0.0) + delta for b in means)
    return {"bands": means, "n": len(widgets), "passed": passed}
