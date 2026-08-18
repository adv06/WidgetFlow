# The Credit Algorithm — how it works & what each file does

WidgetFlow assigns **credit**: for one generated widget (JSX) and its target
screenshot, it decides *which source tokens are responsible for which reward
band* (`layout` / `style` / `perceptual` / `legibility`). A training loop can
then weight the policy gradient onto the tokens that caused the visual error.

This document is the full reference: the idea, the algorithm, every file
function-by-function, the robustness audit, and the limits.

---

## 1. The one idea

Credit is defined by **intervention, not inference**:

> A token's credit for a band = **how much that band's score changes when you
> perturb the token and re-render the whole widget.**

No correspondence (we never match a rendered element to a target region — that
worked on only 13–23% of elements). No ontology, no database. If deleting
`bg-[#20392F]` drops the *style* score by 8 and moves nothing else, that token
owns 8 units of style credit — because style is what changed when you touched it.
A knockout experiment: disable one gene, measure the whole organism.

---

## 2. The algorithm

```
credit_widget(code, scorer):
  1. trace    = scorer.trace(code)                # 1 render → DOM + source spans
  2. dead     = dead_value_map(resolve_dead(code)) # DOM pass → tokens that can't matter
  3. elements = spans(trace)                        # visible, source-mapped tags
  4. base     = scorer(code)                        # baseline band scores
  5. for each element, tokens =
        utility_tokens ∪ attr_tokens ∪ text_tokens  # candidates
  6. for each token t NOT in dead:
        patched = code[:t.start] + t.replacement + code[t.end:]
        got     = scorer(patched)                   # 1 render + score
        Δ_b(t)  = |got_b − base_b|, kept if ≥ min_delta
  7. score(t) = Σ_b  w_b · Δ_b(t) / Σ_t' Δ_b(t')     # normalise per band
  return {tokens sorted by score, counts, n_pruned}
```

- **Step 2** is a *cost* optimisation only — never touches the credit math.
- **Step 6** is the credit. `t.replacement` is `""` for a deletion (class
  utility, inline style, opaque attribute) or a same-shape value (SVG
  colour/number, text). A failed render (`None`) is skipped — absence of
  evidence, never wrong evidence.
- **Step 7** makes each band spend exactly its own weight. Invariant:
  `Σ_t score(t) = Σ_b w_b` over the bands that moved.

---

## 3. The files

```
credit/
  credit.py    orchestration — BandScorer, spans, credit_widget      (the entry point)
  resolve.py   the computed-style dead-token filter (DOM ablation)
  ablate.py    tokenizers + credit_tokens (the render-ablation loop)
  __init__.py  exports
  test_credit.py   the 14-check suite
```

Two different "ablations" live here — do not conflate them:

| | **DOM ablation** (`resolve.py`) | **render ablation** (`ablate.py`) |
|---|---|---|
| changes | a live DOM node's attribute | the source string, then recompiles |
| reads | `getComputedStyle` (stops before paint) | rendered pixels → band scores |
| cost | ~2.5 ms | ~85–700 ms |
| job | prune tokens that can't matter | **assign the credit** |

---

### `credit.py` — orchestration & scoring

**`class BandScorer`** — turns a JSX string into `{band: score vs target}`, and
holds one browser open for the whole widget.
- *Why it holds the browser open:* a fresh context per render is ~677 ms, ~87% of
  it setup/load/Babel/settle, none of it paint. Reusing the page → ~85 ms. So it
  is a context manager (`with BandScorer(gt) as scorer:`).
- `__init__(gt_path, skip=("legibility",), use_cuda=False, scorer=None)` — loads
  the target once; `skip` drops legibility by default (OCR is expensive).
- `__call__(code) -> {band: score} | None` — **this is the `render_score` the
  credit loop calls.** Renders `code`, scores it against the target, returns a
  band dict, or **`None` on any failure** (including a Babel error from a broken
  edit). The loop treats `None` as "no evidence".
- `trace(code)` — a baseline `render_with_trace`, returning DOM elements with
  their `data-w2c-src` source spans.
- *Gotcha:* `_default_scorer` puts `Widget2Code4.0` on `sys.path`, whose
  `generator/` can shadow the 3.0 `JSXRenderer` (a different renderer, no
  `_build_html`). Pass `scorer=` to avoid it.

**`spans(trace) -> [{idx, tag, start, end}]`** — every visible, source-mapped
element. Skips `display:none`/`visibility:hidden`/`opacity≈0` and malformed
spans. `code[start:end]` is the element's **opening tag** — proven to end in `>`,
and opening-tag spans never overlap, so an element's span excludes its children
(no masking needed).

**`credit_widget(code, scorer, *, band_weights, min_delta, max_tokens, prefilter=True, verbose)`**
— the full pipeline. With `prefilter=True`: `resolve_dead` once → dead-map →
tokenise off the trace → `credit_tokens`. Returns `credit_tokens`' dict.

---

### `resolve.py` — the computed-style dead-token filter (DOM ablation)

Only a **cost** optimisation: find tokens that provably cannot change a pixel, so
their render is skipped. It never assigns credit.

**Why dead tokens exist.** A token affects output only if it **wins its CSS
property** against defaults, overrides, specificity, and the flex/grid engine.
Two failure modes, both measured on real code (~9–18% of tokens):
- *restates a default* — `flex-row` when `row` is already the default direction.
- *overridden / layout-constrained* — a real value the engine overrules, e.g.
  `w-[120px]` on an element the flex parent sizes to 40 px. Deleting it changes
  nothing; a source-level analysis can never see this.

**`_JS`** — one `page.evaluate` over every `[data-w2c-src]` element. Per token:
`before = getComputedStyle(el + ≤12 descendants)` → ablate it (`remove class` /
`removeProperty` / `removeAttribute`) → `after` → the set of changed properties →
**restore the exact original string**. Empty change-set ⇒ dead. Three details:
- *snapshot the subtree*, because a parent's `flex` reshapes its children.
- *restore the exact attribute string*, not `classList.add` (which re-orders and
  can flip an equal-specificity rule, poisoning the next token's baseline).
- capped at 12 descendants (`CAP`).

**`resolve_dead(renderer, code, w, h) -> {live, dead, n, trace, png}`** — renders
once with trace, runs `_JS`, splits tokens into `live`/`dead`, each
`{src, kind, value, css}`.

**`dead_value_map(resolved) -> {element_src -> set(values)}`** — the prune set
`credit_tokens` consults, keyed by `(element, value)`. **The soundness gate:**
only `kind ∈ {"utility", "style_decl"}` are prunable. *SVG/host attributes are
never pruned* — `getComputedStyle` does not capture SVG geometry, so removing
`width="36"` collapses the drawing while the style-diff reads empty. Measured:
attributes gave **26% false-dead** (worst 17.7 M pixels); restricting to
utility/style_decl gave **0% over 55 tokens**. A value live *anywhere* on an
element is discarded from its dead-set (only prune what is proven dead).

**Soundness.** For CSS classes/styles the render is a pure function of computed
style, so `no computed-property change ⇒ identical render ⇒ zero band delta` —
verified by rendering every pruned token and diffing pixels (0 false-dead). Open
holes (all fail toward *false-dead* = a wasted render, never wrong credit): the
12-descendant cap; **flex siblings** (a resized sibling isn't in the mutated
element's subtree — close by snapshotting the parent subtree); pseudo-elements;
`hover:`/`focus:` variants (pruned, but correctly, for a static screenshot).

---

### `ablate.py` — tokenizers + the render-ablation credit loop

Four functions. The three tokenizers find candidates; `credit_tokens` runs the
measurement.

**`utility_tokens(code, start, end)`** — the primary tokenizer.
- Every whitespace-delimited chunk of `className="…"` → one token, `replacement=""`
  (**delete**). Exhaustive by construction, 100% coverage, no vocabulary.
- Every comma-separated `key:value` of `style={{…}}` → one token, **delete** — but
  the deletion **swallows a comma** (the trailing one, or the leading one if
  last), because `{{, height }}` is a syntax error. Verified 0/124 breaks.
- Deleting a class utility is *always* valid (removing a substring from a string
  literal), which is why deletion is the default operation.

**`attr_tokens(code, start, end)`** — every attribute except `className`/`style`
(and `data-w2c-src`). Types by the **value's shape** via `_shape_of`, not the
attribute name:
- recognised **colour** (`#hex`, `rgb(...)`, `none`, `currentColor`) or **number**
  (`24`, `18px`) → **swap** a same-shape value (this is what makes SVG `fill`/
  `stroke`/`width` creditable — the 23% of attribute characters `utility_tokens`
  can't see).
- opaque value (a path's `d`, a `viewBox`) → **delete the whole attribute**
  (always valid).

**`_shape_of(value) -> (kind, replacement)`** — the *only* typing in the codebase.
`COLOUR_SHAPE` / `NUMBER_SHAPE` are closed grammars, so they match *every* colour
and number including arbitrary ones (`#9F1C7B`, `137px`) with no database. No
match → `(None, None)` → the caller deletes instead.

**`text_tokens(code, el)`** — the text between an element's `>` and the next `<`.
Bails on `{` (a JSX expression, not literal copy). **Swap** to `"Lorem ipsum
dolor"` — the only channel that reaches **legibility** (word content). (Deletion
would test "does *any* text matter"; the swap tests "are *these words* right".)

**`credit_tokens(code, elements, render_score, *, band_weights, min_delta=0.5, max_tokens=None, dead=None, verbose=False)`**
— the loop and the math.
1. `base = render_score(code)`; `None` ⇒ empty result dict (consistent type).
2. Gather `utility ∪ attr ∪ text` per element, tag each with its owner `src`.
3. **Prune**: drop any candidate whose `(owner, value)` is in `dead` — no render.
4. **Ablate**: splice `replacement`, `render_score`, take absolute per-band delta,
   keep bands ≥ `min_delta`. A `None` render (broken edit) is skipped.
5. **Normalise**: `score(t) = Σ_b w_b · Δ_b(t) / Σ_t' Δ_b(t')`. Each band spends
   its own weight; the loudest band (layout deltas ~6× style's) can't dominate.
6. Return `{tokens (sorted), counts, baseline, n_candidates, n_pruned}`.

---

## 4. How one call executes (data flow)

```
credit_widget(code, scorer)
  └─ resolve_dead → _JS in the browser        →  dead-value map        [1 render]
  └─ scorer.trace → data-w2c-src spans         →  elements
  └─ scorer(code) → render+score               →  base bands           [1 render]
  └─ per element: utility_tokens/attr_tokens/text_tokens  →  candidates
  └─ credit_tokens:
       prune dead
       per token: patch source → scorer(patched) → band deltas         [N renders]
       normalise per band
  └─ {tokens: [{start,end,value,kind,bands,score}], counts, n_pruned}
```

---

## 5. Robustness — `test_credit.py` (14/14)

Run `RENDER=1 python -m credit.test_credit`.

**Unit (10):** tokenizer finds every class utility (42/42); spans never overlap;
deletions offset-consistent; style-decl deletion leaves no dangling comma (0/2
malformed); `_shape_of` types unseen `#9F1C7B`/`137px`, opaque path untyped
(**no database**); normalisation `Σ scores = Σ weights` (0.8000); unmoved band
spends nothing; baseline-fail returns a dict.

**Render (4):** determinism (identical ablation → identical score); **colour →
STYLE** (Δ 8.87); **padding → LAYOUT** (Δ −5.03); dead-filter **0 false-dead**.

*Caveat:* the suite's soundness check is vacuous on the clean test widget (no
computed-style-dead tokens there). The substantive evidence is a separate run
over 6 rollout widgets — **55 pruned, 0 false-dead** — after fixing the SVG bug.

---

## 6. Is it robust? & the limits

**The mechanism is robust** — pure measurement, no assumption to be wrong;
proven deterministic, correctly routing, non-breaking (0/124), weight-preserving.
**The dead-filter is sound for what it prunes** (0 false-dead), after the SVG
fix, with the sibling/deep-subtree holes still open.

Two ceilings the algorithm cannot exceed:

1. **Cost is O(tokens × pixels).** ~711 ms/token; ~5.7 h/DAPO-pass raw, ~0.7–1.6 h
   with persistent-page + dead-filter + below-mean-only. **Too costly as a naive
   per-pass in-loop signal** — best used offline (credit maps) or to precompute a
   reusable `token → band` table.
2. **Credit is only as fine as the reward metric.** Proven: deleting an inner
   `flex` reflowed 1.37 M pixels yet moved the layout *metric* by 0.00, so it
   reads dead. Correct *as reward credit*; not a claim about token type. Token
   *type* (flex → layout) is a different question, answerable only by DOM
   property resolution, not ablation.

Smaller: `text_tokens`/`attr_tokens` still substitute (the two non-deletion
spots); legibility is skipped by default (OCR cost); `training/` is unbuilt.
