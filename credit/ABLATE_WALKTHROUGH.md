# `ablate.py` — a block-by-block walkthrough

This file is the **render-ablation credit loop**. Its one job: given a piece of
generated JSX and a scorer, decide **which source tokens are responsible for which
reward band** — not by guessing from the source text, but by *intervention*:
delete or swap a token, re-render the whole widget, and measure how each band's
score moved. A token that, when removed, drops the *style* score by 8 owns 8 units
of style credit.

The file has **four functions** and a few module-level regexes:

| Function | Role |
|---|---|
| `text_tokens` | tokenizer — the literal text between tags |
| `utility_tokens` | tokenizer — each Tailwind class utility + each inline-style declaration |
| `attr_tokens` | tokenizer — every other attribute (SVG `fill`, `width`, `d`, …) |
| `credit_tokens` | the loop — gather candidates, prune, ablate, score |

Every token the tokenizers emit is a **plain dict** with the same shape:

```python
{"start": int, "end": int, "kind": str, "value": str, "replacement": str}
```

- `start`/`end` are **absolute character offsets** into the full `code` string.
- `replacement` is what to splice in: `""` means **delete** `code[start:end]`; a
  non-empty string means **swap** it for a same-shaped value.
- `kind` is a label (`utility`, `style_decl`, `attr_colour`, `attr_number`,
  `attr_drop`, `text`) used only for reporting.

The snippets below appear **in file order** — concatenated (with the usual two
blank lines between top-level blocks) they reproduce the whole file.

---

## 1. Imports

```python
from __future__ import annotations
import re
```

`from __future__ import annotations` makes the type hints (`int | None`) strings at
runtime, so they cost nothing and don't need the types imported. `re` is the only
dependency — all the tokenizing is regex + string slicing. No rendering happens
here; that's injected as a callback (`render_score`).

---

## 2. `text_tokens` — the literal copy between tags

```python
def text_tokens(code: str, el, next_tag_start: int | None = None):
    a = el["end"]
    b = code.find("<", a)
    if b < 0 or b <= a:
        return []
    raw = code[a:b]
    if not raw.strip() or "{" in raw:
        return []
    lead = len(raw) - len(raw.lstrip())
    return [{"start": a + lead, "end": a + len(raw.rstrip()), "kind": "text",
             "value": raw.strip(), "replacement": ""}]
```

`el` is one **element span** (`{idx, tag, start, end}`) produced by `spans()` in
`credit.py`; `el["end"]` is the offset just past the element's **opening tag** —
i.e. right after the `>`.

- `a = el["end"]` — start looking at the character after `>`.
- `b = code.find("<", a)` — the next `<` is where the text ends (either a child tag
  or this element's closing `</…>`). If there's no `<` after, or it's not past `a`,
  there's no text → `[]`.
- `raw = code[a:b]` — the candidate text, e.g. `"Balance"` in `<span>Balance</span>`.
- **Two bail-outs:** empty/whitespace-only text is nothing to credit; and if `raw`
  contains `{`, it's a **JSX expression** (`<span>{count}</span>`), not literal copy —
  we can't treat it as text, so skip.
- `lead` counts leading whitespace so the token's span covers **only the trimmed
  text**, not the surrounding indentation. `start = a + lead`,
  `end = a + len(raw.rstrip())`.
- `replacement": ""` → the operation is **deletion**. Ablating text means removing
  it and re-scoring; if a band moves, that text mattered to that band. (Deletion
  answers *"does this text matter?"*; it does not test *which* words are right.)

`next_tag_start` is an unused optional parameter kept for call-site compatibility.

---

## 3. `credit_tokens` — the measurement loop

This is the heart of the file. It takes the code, the element spans, and a
`render_score` callback (the `BandScorer`), and returns the credited tokens. We
walk it in six contiguous pieces.

### 3a. Baseline

```python
def credit_tokens(code, elements, render_score, *, band_weights,
                  min_delta=0.5, max_tokens=None, dead=None, verbose=False):

    base = render_score(code)
    if base is None:
        return {"tokens": [], "counts": {}, "baseline": None, "n_candidates": 0,
                "n_pruned": 0}
    dead = dead or {}
    out = []
    cands = []
```

- `render_score(code)` renders the **unmodified** widget and returns the band dict,
  e.g. `{"layout": 34.6, "style": 44.8, "perceptual": 74.0}`. This is the **baseline**
  every ablation is compared against.
- If the baseline render fails (`None`), there's nothing to compare to — return an
  empty result with a consistent shape (so callers never crash on a missing key).
- `dead` is the optional **prune set** (`{owner -> {values}}`) from `resolve.py`;
  `dead or {}` makes "not supplied" behave as "prune nothing".
- `out` collects credited tokens; `cands` collects all candidates before ablation.

`band_weights` is keyword-only (after `*`) — the caller must pass it explicitly.

### 3b. Gather candidates from every element

```python
    for el in elements:
        src = f'{el["start"]}:{el["end"]}'


        toks = (utility_tokens(code, el["start"], el["end"])
                + attr_tokens(code, el["start"], el["end"])
                + text_tokens(code, el))
        for t in toks:
            t["owner"] = src
            cands.append(t)
    n_all = len(cands)
```

For each element we run all three tokenizers over that element's opening-tag span
(`utility_tokens` and `attr_tokens` take `start, end`; `text_tokens` takes the whole
`el`). Their token lists are concatenated.

- `src = "start:end"` is a **stable string id for the owning element**. Each token
  is tagged with `t["owner"] = src`. That's the key the dead-map is keyed on
  (`dead[owner]`), so pruning can say "value X is dead *on this element*".
- `n_all` records the candidate count **before** pruning (for the report).

### 3c. Prune, sort, cap

```python
    cands = [t for t in cands if t["value"] not in dead.get(t["owner"], ())]
    n_pruned = n_all - len(cands)
    cands.sort(key=lambda d: d["start"])
    if max_tokens:
        cands = cands[:max_tokens]
```

- **Prune:** drop any candidate whose `(owner, value)` is in the dead set — those
  tokens were proven (by the computed-style pass in `resolve.py`) to change no pixel,
  so ablating them would waste a render. `dead.get(owner, ())` defaults to an empty
  tuple, so unknown owners prune nothing.
- `n_pruned` = how many we skipped this way.
- **Sort by `start`** so the report reads top-to-bottom in source order.
- **Cap** at `max_tokens` if given — a cost lever (each remaining candidate costs one
  render). Note: after this line `len(cands)` is the number we will actually ablate.

### 3d. Ablate each candidate and measure the deltas

```python
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
            out.append({**t, "bands": moved})
```

This is the intervention:

- `patched = code[:start] + replacement + code[end:]` — splice the edit in. For a
  deletion (`replacement=""`) this cuts the token out; for a swap it substitutes.
  Because offsets are absolute, this is a single clean slice.
- `got = render_score(patched)` — **re-render and re-score the patched widget.**
- **Broken edit ⇒ skip.** If the patch produced un-renderable JSX (a Babel error →
  `None`), we `continue`. This is *absence of evidence*, never wrong evidence — a
  failed render never counts as credit.
- **Per-band delta:** for each baseline band `v0`, `d = |got[b] − v0|`. We keep only
  bands that moved at least `min_delta` (default 0.5) — a noise floor, so
  sub-half-point jitter isn't mistaken for real credit. `got.get(b, v0)` guards a
  band missing from the patched score (treats it as unchanged).
- `verbose` prints one aligned line per token showing which bands moved (or
  `(no band moved)`).
- If **any** band moved, we append a copy of the token with a new `"bands"` field:
  `{**t, "bands": moved}` (e.g. `{... , "bands": {"style": 43.0, "layout": 12.3}}`).

After this loop, `out` holds only the tokens that actually changed a band.

### 3e. Normalise credit per band

```python
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
```

Now we turn raw deltas into a single **score** per token.

- `totals[b]` = the **sum of all deltas** on band `b` across every credited token.
  This is the total "movement" band `b` saw.
- `counts[b]` = how many tokens touched band `b` (reporting only).
- **The score formula** (per token `t`):

  ```
  score(t) = Σ_b  band_weights[b] · ( d_b(t) / totals[b] )
  ```

  For each band the token moved, take its **share** of that band's total movement
  (`d_b / totals[b]`, a fraction in `[0,1]`), weight it by the band's importance
  (`band_weights[b]`), and sum across bands. The `if totals.get(b) > 0` guard avoids
  divide-by-zero.

  **Why divide by the band total?** It makes each band spend exactly its own weight:
  summing `score(t)` over all tokens gives `Σ_b band_weights[b]` (over bands that
  moved). So a "loud" band (layout deltas are ~6× style's in raw points) can't
  dominate — every band contributes the same total budget, split among *its*
  responsible tokens. A token that owns all of style's movement gets style's full
  weight; a token that owns half gets half.

### 3f. Sort and return

```python
    out.sort(key=lambda d: -d["score"])
    if verbose:
        print("\n  tokens affecting each band: "
              + "  ".join(f"{b}={counts.get(b, 0)}" for b in band_weights))
    return {"tokens": out, "counts": counts, "baseline": base,
            "n_candidates": len(cands), "n_pruned": n_pruned}
```

- Sort tokens by **descending score** — most-responsible first.
- `verbose` prints the per-band token counts.
- Return the bundle: `tokens` (sorted, each with `start/end/kind/value/bands/score`),
  `counts`, the `baseline` bands, `n_candidates` (how many were ablated, i.e. post-cap),
  and `n_pruned`. This dict is what `credit_widget` returns and what
  `credit_weight.rollout_token_weights` consumes to build per-completion-token weights.

---

## 4. Class/style regexes

```python
CLASS_ATTR = re.compile(r'className\s*=\s*"([^"]*)"')
STYLE_ATTR = re.compile(r"style\s*=\s*\{\{([^}]*)\}\}")
```

- `CLASS_ATTR` captures the inside of `className="…"` — group 1 is the class string.
- `STYLE_ATTR` captures the inside of `style={{…}}` — group 1 is the declaration
  list. (`[^}]*` stops at the first `}`, which is fine for flat style objects.)

Both are module-level so they compile once.

---

## 5. `utility_tokens` — classes and inline-style declarations

```python
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
```

`seg = code[start:end]` is the element's **opening tag** only. Offsets inside `seg`
are local, so every emitted offset is rebased with `start + …` back to the full string.

**Classes:** for each `className="…"`, `base = m.start(1)` is where the class string
begins inside `seg`. `re.finditer(r"\S+", body)` splits the class string on
whitespace — each run of non-space characters is one utility (`flex`, `flex-col`,
`p-4`, `bg-[#20392F]`). Each becomes a token:

- `start = start + base + um.start()` — absolute offset of this utility.
- `replacement": ""` — **delete**. Removing one whitespace-delimited chunk from a
  string literal is *always* valid JSX, which is why deletion is the default here.
- This is exhaustive by construction — every class, no vocabulary needed.

```python
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
```

**Inline styles:** `body` is the text inside `style={{ … }}`. `re.finditer(r"[^,]+")`
splits on commas; keeping only pieces containing `:` filters out empties and keeps
real declarations like `width: '120px'`.

The tricky part is **comma handling**. Deleting just `width: '120px'` from
`{{ width: '120px', height: '40px' }}` would leave `{{ , height: '40px' }}` — a
**syntax error** (Babel fails, render returns `None`, and the token is silently
dropped as "no evidence"). Unlike a class utility, an object literal's separators
must be maintained. So the span is widened to **swallow a comma**:

- `after = body.find(",", b)` — is there a comma *after* this declaration? If so,
  extend `b` past it (`after + 1`): delete `width: '120px',`.
- Otherwise (this is the last declaration) `before = body.rfind(",", 0, a)` — extend
  `a` back to the *preceding* comma: delete `, height: '40px'`.

Either way the remaining object stays valid. `value` is the trimmed declaration text;
`kind="style_decl"`; `replacement=""` (delete). Finally sort by offset.

---

## 6. Attribute regexes and shape grammars

```python
ATTR = re.compile(r'([A-Za-z_:][\w:.-]*)\s*=\s*(?:"([^"]*)"|\{([^{}]*)\})')
COLOUR_SHAPE = re.compile(r"^\s*(#[0-9a-fA-F]{3,8}|(?:rgb|rgba|hsl|hsla)\([^)]*\)"
                          r"|white|black|currentColor|none|transparent)\s*$", re.I)
NUMBER_SHAPE = re.compile(r"^\s*(-?\d+(?:\.\d+)?)(px|rem|em|%|)\s*$")
```

- `ATTR` matches **any attribute**: `name="value"` (group 2) *or* `name={expr}`
  (group 3). The name (group 1) allows letters, `_`, `:`, digits, `.`, `-` (covers
  `data-w2c-src`, `viewBox`, `stroke-width`, etc.).
- `COLOUR_SHAPE` / `NUMBER_SHAPE` are **closed grammars** — anchored (`^…$`) patterns
  that recognise *any* CSS colour or number, including arbitrary ones like `#9F1C7B`
  or `137px`, with no lookup table. A colour is a hex, an `rgb()/hsl()` call, or a
  keyword; a number is an optional-sign decimal with an optional unit.

---

## 7. `_shape_of` — type a value by its shape

```python
def _shape_of(value: str):

    if COLOUR_SHAPE.match(value):
        v = value.strip().lower()
        return "colour", ("#0E7A9C" if v in ("#5a2e10",) else "#5A2E10")
    m = NUMBER_SHAPE.match(value)
    if m:
        n, unit = float(m.group(1)), m.group(2)
        return "number", f"{max(1, int(round(n * 1.6))) if n else 6}{unit}"
    return None, None
```

This is the **only place typing happens**, and it types by what the value *looks
like*, never by the attribute name. It returns `(kind, replacement)`:

- **Colour** → swap to a fixed different colour `#5A2E10`. The guard
  (`"#0E7A9C" if v == "#5a2e10"`) handles the one case where the value *already is*
  `#5A2E10`, swapping to a different colour instead — otherwise the "swap" would be a
  no-op and the token would read as dead.
- **Number** → swap to `round(n * 1.6)` in the **same unit** (so `36` → `58`,
  `18px` → `29px`); if the number is `0`, use `6` (scaling zero gives zero, another
  no-op). Multiplying by 1.6 guarantees a visibly different value.
- **No match** → `(None, None)`; the caller falls back to deleting the whole
  attribute.

Why swap instead of delete? For a colour or geometry number, **deletion destroys the
element** (removing `width="36"` collapses an SVG rect), which tells you the attribute
*exists* but not whether *this value* is right. Substituting a same-shaped value
isolates the value itself. Substitution needs a type — hence it's only available where
the shape is recognised.

---

## 8. `attr_tokens` — every other attribute

```python
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
```

Walks every attribute in the opening tag (`seg`), skipping:

- `data-w2c-src` — the renderer's own source-map attribute, not part of the widget.
- `className` / `style` — already handled at finer granularity by `utility_tokens`.

For each remaining attribute:

- `quoted` distinguishes `name="value"` (group 2) from `name={expr}` (group 3).
- `value` is the string inside; `vs` is the offset of that value within `seg`.
- `_shape_of(value)` types it.
- **`if kind and quoted`** → the value has a recognised colour/number shape *and* is
  in double quotes → emit a **swap token** covering just the value:
  `kind = "attr_colour"` or `"attr_number"`, `replacement = rep` (the substitute).
  This is what makes SVG `fill="#20392F"` / `width="36"` creditable — the attribute
  characters `utility_tokens` can't see.
- **`else`** → an opaque value (a path `d="M12 3…"`, a `viewBox`), a brace expression,
  or a shape we can't type → **delete the whole attribute** (`kind="attr_drop"`,
  span = `m.start()..m.end()`, `replacement=""`). Deleting an entire `name="value"`
  from a tag is always valid syntax. `value` here is a truncated label
  (`name=firstchars`) for the report, not the operation.

Finally sort by offset and return.

---

## How the pieces connect

`credit.py:credit_widget` calls `credit_tokens(code, elements, scorer, …)`, passing
element spans from `spans(trace)` and the `BandScorer` as `render_score`. Inside,
`credit_tokens` fans out to these three tokenizers per element, prunes with the
dead-map from `resolve.py`, then ablates and scores. The returned `tokens` list
(each `{start, end, kind, value, bands, score}`) is later mapped onto a rollout's
actual completion tokens by `credit_weight.py`, and those per-token weights multiply
the DAPO loss so the policy gradient concentrates on the tokens that caused the
visual error.
