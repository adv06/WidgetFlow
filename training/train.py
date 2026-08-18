"""End-to-end WidgetFlow training: Qwen3-VL-8B-Instruct + LoRA, DAPO on the
widget2code-benchmark train split. Wires every box of the architecture:

  TrainFilter(130) -> Policy(N=8 rollouts) -> render_and_score -> dapo_loss(step)
                   -> metric_probe(60 held-out) -> Controller -> rotation -> swap
                   -> stop when every band >= baseline + delta.

Self-contained: model log-probs via the vendored model/loader.py (correct Qwen3-VL
M-RoPE), rendering via the vendored generator/, scoring via the vendored reward/.

Run (does a real multi-hour run; pick GPUs deliberately):
  CUDA_VISIBLE_DEVICES=4 python -m training.train --passes 20 --cache data/wtargets
"""
from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import torch
from PIL import Image

from training.policy import Policy
from training.train_filter import TrainFilter
from training.rotation import to_drop
from training.controller import Controller
from training.render_and_score import render_and_score
from training.dapo import dapo_loss
from training.metric_probe import metric_probe
from training.credit_weight import credit_weights

from reward.scorer import score_bands

DATASET = "Djanghao/widget2code-benchmark"


def materialize(cache: Path, limit=0):
    """Save train-split images to PNGs once; return [png_path] by index. limit>0
    streams only the first `limit` rows (smoke tests, no full download)."""
    from datasets import load_dataset
    cache.mkdir(parents=True, exist_ok=True)
    if limit:
        ds = load_dataset(DATASET, split="train", streaming=True)
        rows = (r for i, r in zip(range(limit), ds))
    else:
        rows = load_dataset(DATASET, split="train")
    paths = []
    for i, row in enumerate(rows):
        p = cache / f"w{i}.png"
        if not p.exists():
            row["image"].convert("RGB").save(p)
        paths.append(str(p))
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/wtargets", help="dir for materialized target PNGs")
    ap.add_argument("--passes", type=int, default=20)
    ap.add_argument("--K", type=int, default=8, help="rollouts per widget")
    ap.add_argument("--train_size", type=int, default=130)
    ap.add_argument("--val_size", type=int, default=60)
    ap.add_argument("--rotate", type=int, default=30, help="widgets swapped per pass")
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--delta", type=float, default=0.02, help="stop-rule margin over baseline")
    ap.add_argument("--max_widgets", type=int, default=0, help="cap widgets/pass (0 = all)")
    ap.add_argument("--limit", type=int, default=0, help="stream only first N widgets (smoke)")
    ap.add_argument("--credit", type=int, default=1, help="1=credit-weight the DAPO loss")
    ap.add_argument("--credit_max_tokens", type=int, default=40, help="ablations/rollout cap")
    ap.add_argument("--credit_alpha", type=float, default=3.0, help="peak token upweight")
    ap.add_argument("--max_new_tokens", type=int, default=2048, help="rollout length cap")
    ap.add_argument("--grad_ckpt", type=int, default=1, help="gradient checkpointing (0=faster gen)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    png = materialize(Path(args.cache), limit=args.limit)
    def image_of(i):
        return Image.open(png[i]).convert("RGB")

    # 60 held-out for the probe; the rest is the rotating training pool.
    val_idx = list(range(args.val_size))
    pool = list(range(args.val_size, len(png)))
    val = [(image_of(i), png[i]) for i in val_idx]

    policy = Policy.load(lora=True, gradient_checkpointing=bool(args.grad_ckpt),
                         max_new_tokens=args.max_new_tokens)
    opt = torch.optim.AdamW([p for p in policy.model.parameters() if p.requires_grad], lr=args.lr)

    tf = TrainFilter(pool, size=args.train_size, seed=args.seed)
    ctrl = Controller()

    baseline = metric_probe(policy, val, score_fn=score_bands)["bands"]
    print(f"[baseline] {fmt(baseline)}")

    for p in range(args.passes):
        varmap, pending = {}, 0
        widgets = tf.training_set[: args.max_widgets or None]
        for idx in widgets:
            rollouts = policy.rollout(image_of(idx), n=args.K)
            scored = render_and_score(png[idx], rollouts, weights=ctrl.w, score_fn=score_bands)
            varmap[idx] = [s.reward for s in scored]                 # rotation ranks on this
            credit = (credit_weights(scored, policy.tok, png[idx], alpha=args.credit_alpha,
                                     max_tokens=args.credit_max_tokens, score_fn=score_bands)
                      if args.credit else {})                        # Credit Weighting box
            loss = dapo_loss(policy, scored, credit=credit)
            if loss is None:                                         # no reward spread -> skip
                continue
            (loss / args.grad_accum).backward()
            pending += 1
            if pending == args.grad_accum:
                opt.step(); opt.zero_grad(); pending = 0
        if pending:
            opt.step(); opt.zero_grad()

        report = metric_probe(policy, val, baseline=baseline, delta=args.delta, score_fn=score_bands)
        ctrl.update(report["bands"])
        print(f"[pass {p}] bands={fmt(report['bands'])} weights={fmt(ctrl.w)} "
              f"passed={report['passed']}")
        if report["passed"]:
            print("[stop] all bands >= baseline + delta -> run eval on 1000")
            break
        tf.swap(to_drop(varmap, args.rotate))                       # rotation -> update pool


def fmt(d):
    return {k: round(v, 3) for k, v in d.items()}


if __name__ == "__main__":
    main()
