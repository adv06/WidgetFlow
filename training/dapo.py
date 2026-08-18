import torch


def advantages(rewards, eps=1e-6):

    r = torch.as_tensor(rewards, dtype=torch.float32)
    return (r - r.mean()) / (r.std(unbiased=False) + eps)


def dapo_loss(policy, scored, *, clip_low=0.2, clip_high=0.28, std_threshold=0.0,
              credit=None):
    r = torch.tensor([x.reward for x in scored], dtype=torch.float32)
    if r.std(unbiased=False) <= std_threshold:
        return None
    adv = (r - r.mean()) / (r.std(unbiased=False) + 1e-6)

    num, ntok = 0.0, 0
    for a, x in zip(adv, scored):
        ro = x.rollout
        ratio = torch.exp(policy.logprob(ro) - ro.old_logprob)
        a = a.to(ratio.device)
        per_tok = -torch.min(ratio * a, ratio.clamp(1 - clip_low, 1 + clip_high) * a)
        if credit is not None:
            per_tok = per_tok * credit.get(ro, 1.0)
        num = num + per_tok.sum()
        ntok += per_tok.numel()
    return num / max(ntok, 1)
