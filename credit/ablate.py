from __future__ import annotations
import re

def text_tokens(code: str, el, next_tag_start: int | None = None):
    a = el["end"]
    b = code.find("<", a) # by construction we cannot contain any tag 
    if b < 0 or b <= a:
        return []
    raw = code[a:b]
    if not raw.strip() or "{" in raw:
        return []
    lead = len(raw) - len(raw.lstrip())
    return [{"start": a + lead, "end": a + len(raw.rstrip()), "kind": "text",
             "value": raw.strip(), "replacement": ""}]


def credit_tokens(code, elements, render_score, *, band_weights,
                  min_delta=0.5, max_tokens=None, dead=None, verbose=False):

    base = render_score(code)
    if base is None:
        return {"tokens": [], "counts": {}, "baseline": None, "n_candidates": 0,
                "n_pruned": 0}
    dead = dead or {}
    out = []
    cands = []
    for el in elements:
        src = f'{el["start"]}:{el["end"]}'


        toks = (utility_tokens(code, el["start"], el["end"])
                + attr_tokens(code, el["start"], el["end"])
                + text_tokens(code, el))
        for t in toks:
            t["owner"] = src
            cands.append(t)
    n_all = len(cands)
    cands = [t for t in cands if t["value"] not in dead.get(t["owner"], ())]
    n_pruned = n_all - len(cands)
    cands.sort(key=lambda d: d["start"])
    if max_tokens:
        cands = cands[:max_tokens]

    for t in cands:
        patched = code[:t["start"]] + t["replacement"] + code[t["end"]:]
        got = render_score(patched)
        if got is None:
            continue
        moved = {}
        for b, v0 in base.items():
            d = abs(float(got.get(b, v0)) - float(v0))
            if d >= min_delta:
                moved[b] = d
        if verbose:
            print(f"    {t['kind']:<15} {t['value'][:18]:<20} -> "
                  + (", ".join(f"{b} {d:+.1f}" for b, d in moved.items()) or "(no band moved)"))
        if moved:
            out.append({**t, "bands": moved}) # important

    totals: dict[str, float] = {}
    for t in out:
        for b, d in t["bands"].items():
            totals[b] = totals.get(b, 0.0) + d
    counts: dict[str, int] = {}
    for t in out:
        for b in t["bands"]:
            counts[b] = counts.get(b, 0) + 1
    for t in out:
        t["score"] = float(sum(band_weights.get(b, 0.0) * d / totals[b]
                               for b, d in t["bands"].items() if totals.get(b, 0.0) > 0))
    out.sort(key=lambda d: -d["score"])
    if verbose:
        print("\n  tokens affecting each band: "
              + "  ".join(f"{b}={counts.get(b, 0)}" for b in band_weights))
    return {"tokens": out, "counts": counts, "baseline": base,
            "n_candidates": len(cands), "n_pruned": n_pruned} # at this  point in time we have all the tokens with their normalized score for band


CLASS_ATTR = re.compile(r'className\s*=\s*"([^"]*)"')
STYLE_ATTR = re.compile(r"style\s*=\s*\{\{([^}]*)\}\}")


def utility_tokens(code: str, start: int, end: int):
    seg = code[start:end]
    out = []
    for m in CLASS_ATTR.finditer(seg):
        body, base = m.group(1), m.start(1)
        for um in re.finditer(r"\S+", body):
            out.append({"start": start + base + um.start(),
                        "end": start + base + um.end(),
                        "kind": "utility", "value": um.group(0),
                        "replacement": ""})
    for m in STYLE_ATTR.finditer(seg):
        body, base = m.group(1), m.start(1)
        decls = [d for d in re.finditer(r"[^,]+", body) if ":" in d.group(0)]
        for i, dm in enumerate(decls):


            a, b = dm.start(), dm.end()
            after = body.find(",", b)
            if after != -1:
                b = after + 1
            else:
                before = body.rfind(",", 0, a)
                if before != -1:
                    a = before
            out.append({"start": start + base + a,
                        "end": start + base + b,
                        "kind": "style_decl", "value": dm.group(0).strip(),
                        "replacement": ""})
    out.sort(key=lambda d: d["start"])
    return out


ATTR = re.compile(r'([A-Za-z_:][\w:.-]*)\s*=\s*(?:"([^"]*)"|\{([^{}]*)\})')
COLOUR_SHAPE = re.compile(r"^\s*(#[0-9a-fA-F]{3,8}|(?:rgb|rgba|hsl|hsla)\([^)]*\)"
                          r"|white|black|currentColor|none|transparent)\s*$", re.I)
NUMBER_SHAPE = re.compile(r"^\s*(-?\d+(?:\.\d+)?)(px|rem|em|%|)\s*$")


def _shape_of(value: str):

    if COLOUR_SHAPE.match(value):
        v = value.strip().lower()
        return "colour", ("#0E7A9C" if v in ("#5a2e10",) else "#5A2E10")
    m = NUMBER_SHAPE.match(value)
    if m:
        n, unit = float(m.group(1)), m.group(2)
        return "number", f"{max(1, int(round(n * 1.6))) if n else 6}{unit}"
    return None, None


def attr_tokens(code: str, start: int, end: int, *, skip=("data-w2c-src",)):

    seg = code[start:end]
    out = []
    for m in ATTR.finditer(seg):
        name = m.group(1)
        if name in skip:
            continue
        if name == "className" or name == "style":
            continue
        quoted = m.group(2) is not None
        value = m.group(2) if quoted else (m.group(3) or "")
        vs = m.start(2) if quoted else m.start(3)
        kind, rep = _shape_of(value)
        if kind and quoted:
            out.append({"start": start + vs, "end": start + vs + len(value),
                        "kind": f"attr_{kind}", "value": value, "replacement": rep})
        else:

            out.append({"start": start + m.start(), "end": start + m.end(),
                        "kind": "attr_drop", "value": f"{name}={value[:24]}",
                        "replacement": ""})
    out.sort(key=lambda d: d["start"])
    return out

# notes : ablate.py essentially deletes tokens, computes score, checks the scored does correct weighting 
# uses regex to find attribute, utility tokens, zeros them
