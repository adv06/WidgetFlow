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
    ap.add_argument("--credit_rollouts", type=int, default=2, help="worst-N below-mean rollouts to credit/widget")
    ap.add_argument("--credit_alpha", type=float, default=3.0, help="peak token upweight")
    ap.add_argument("--reward_bands", default=None,
                    help="reward ONLY these bands, equal fixed weight, Controller off (e.g. style,perceptual)")
    ap.add_argument("--max_new_tokens", type=int, default=2048, help="rollout length cap")
    ap.add_argument("--grad_ckpt", type=int, default=1, help="gradient checkpointing (0=faster gen)")
    ap.add_argument("--sft", default=None, help="SFT adapter to init the trainable LoRA from")
    ap.add_argument("--save_dir", default=None, help="save the trained adapter here each pass")
    ap.add_argument("--probe_tokens", type=int, default=1536, help="greedy budget for the held-out probe")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    png = materialize(Path(args.cache), limit=args.limit)
    def image_of(i):
        return Image.open(png[i]).convert("RGB")


    val_idx = list(range(args.val_size))
    pool = list(range(args.val_size, len(png)))
    val = [(image_of(i), png[i]) for i in val_idx]

    policy = Policy.load(adapter=args.sft, lora=True, gradient_checkpointing=bool(args.grad_ckpt),
                         max_new_tokens=args.max_new_tokens)
    opt = torch.optim.AdamW([p for p in policy.model.parameters() if p.requires_grad], lr=args.lr)

    tf = TrainFilter(pool, size=args.train_size, seed=args.seed)
    ctrl = Controller()
    fixed_w = None
    if args.reward_bands:
        sel = {b.strip() for b in args.reward_bands.split(",")}
        fixed_w = {b: (1.0 / len(sel) if b in sel else 0.0) for b in ctrl.w}
        print(f"[reward] fixed bands={sorted(sel)} weights={fmt(fixed_w)} (Controller OFF)", flush=True)

    baseline = metric_probe(policy, val, score_fn=score_bands, max_new_tokens=args.probe_tokens)["bands"]
    print(f"[baseline] {fmt(baseline)}")

    for p in range(args.passes):
        varmap, pending = {}, 0
        widgets = tf.training_set[: args.max_widgets or None]
        for idx in widgets:
            rollouts = policy.rollout(image_of(idx), n=args.K)
            scored = render_and_score(png[idx], rollouts, weights=(fixed_w or ctrl.w), score_fn=score_bands)
            varmap[idx] = [s.reward for s in scored]
            credit = (credit_weights(scored, policy.tok, png[idx], alpha=args.credit_alpha,
                                     max_tokens=args.credit_max_tokens,
                                     max_rollouts=args.credit_rollouts, score_fn=score_bands)
                      if args.credit else {})
            loss = dapo_loss(policy, scored, credit=credit)
            rs = varmap[idx]
            nvalid = sum(1 for s in scored if s.bands)
            print(f"[pass {p}] w{idx} valid {nvalid}/{len(rs)} reward~{sum(rs)/len(rs):.1f} "
                  f"spread {max(rs)-min(rs):.1f} loss={None if loss is None else round(float(loss),4)} "
                  f"credit={len(credit)}", flush=True)
            if loss is None:
                continue
            (loss / args.grad_accum).backward()
            pending += 1
            if pending == args.grad_accum:
                opt.step(); opt.zero_grad(); pending = 0
        if pending:
            opt.step(); opt.zero_grad()

        report = metric_probe(policy, val, baseline=baseline, delta=args.delta, score_fn=score_bands, max_new_tokens=args.probe_tokens)
        if fixed_w is None:
            ctrl.update(report["bands"])
        print(f"[pass {p}] bands={fmt(report['bands'])} weights={fmt(fixed_w or ctrl.w)} "
              f"passed={report['passed']}", flush=True)
        if args.save_dir:
            policy.model.save_pretrained(args.save_dir)
        if report["passed"]:
            print("[stop] all bands >= baseline + delta -> run eval on 1000")
            break
        tf.swap(to_drop(varmap, args.rotate))


def fmt(d):
    return {k: round(v, 3) for k, v in d.items()}


if __name__ == "__main__":
    main()
