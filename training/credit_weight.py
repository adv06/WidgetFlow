import torch

from credit.credit import BandScorer, credit_widget


def token_char_offsets(tok, comp_ids):
    """Char [start, end) of every completion token within decode(comp_ids).

    Returns exactly len(comp_ids) spans, aligned to the ACTUAL sampled ids (not a
    re-tokenisation that might differ). Fast path: if re-tokenising the decoded
    text reproduces the ids, use the tokenizer's own offset mapping. Exact
    fallback: cumulative-decode length per prefix (monotonic in the id sequence).
    """
    ids = [int(t) for t in comp_ids]
    text = tok.decode(ids, skip_special_tokens=True)
    try:
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        if [int(t) for t in enc["input_ids"]] == ids:
            return [(int(a), int(b)) for a, b in enc["offset_mapping"]], text
    except Exception:
        pass
    offs, prev = [], 0
    for i in range(1, len(ids) + 1):
        cur = len(tok.decode(ids[:i], skip_special_tokens=True))
        offs.append((prev, max(prev, cur)))
        prev = max(prev, cur)
    return offs, text


def rollout_token_weights(r, tok, credit_tokens, *, alpha=3.0):
    n = len(r.comp_ids)
    spans = [(int(t["start"]), int(t["end"]), float(t["score"]))
             for t in credit_tokens if int(t["end"]) > int(t["start"])]
    if not spans:
        return torch.ones(n, dtype=torch.float32, device=r.comp_ids.device)
    top = max(s for _, _, s in spans) or 1.0

    offs, _ = token_char_offsets(tok, r.comp_ids)
    w = [1.0] * len(offs)
    for i, (ts, te) in enumerate(offs):
        if te <= ts:
            continue
        best = 0.0
        for a, b, score in spans:
            if ts < b and a < te:                 # char-span overlap
                best = max(best, score / top)
        if best > 0.0:
            w[i] = 1.0 + (alpha - 1.0) * best
    w = (w + [1.0] * n)[:n]                        # keep length == comp_ids
    return torch.tensor(w, dtype=torch.float32, device=r.comp_ids.device)


def credit_weights(scored, tok, gt_path, *, alpha=3.0, max_tokens=None,
                   min_delta=0.5, score_fn=None, verbose=False):
   
    # Only rollouts that actually rendered (non-empty bands) have a visual to
    # ablate; a failed/truncated render is below-mean but has no evidence.
    below = [x for x in scored if x.below_mean and x.bands]
    if not below:
        return {}
    weights = {}
    with BandScorer(gt_path, scorer=score_fn) as scorer:
        for x in below:
            r = x.rollout
            try:
                # prefilter=False: skip the computed-style dead-token pass (a cost
                # optimisation only; per ALGORITHM.md it never changes the credit
                # math). Can be re-enabled now that the vendored renderer has
                # _build_html, but ablating every candidate is simplest and correct.
                res = credit_widget(r.text, scorer, min_delta=min_delta,
                                    max_tokens=max_tokens, prefilter=False, verbose=verbose)
            except Exception:
                continue                     # flaky render -> no evidence, weight stays 1.0
            toks = res.get("tokens", [])
            if toks:
                weights[r] = rollout_token_weights(r, tok, toks, alpha=alpha)
    return weights
