"""Computed-style pre-filter for ablation credit.

Render ablation is the credit signal: perturb a source token, re-render, re-score
against the target, and the band deltas ARE the credit -- measured, not assumed.
No ontology, no property->band table. The reward defines the bands operationally
(a source-level map cannot see that opacity moves SSIM as well as palette), so we
let the render tell us.

The one thing worth doing cheaply first is DEAD-TOKEN PRUNING. A token that
changes no computed style property cannot change a single pixel, so re-rendering
it is wasted. getComputedStyle before/after a DOM-level ablation costs ~2.5ms
against ~85ms for a render, so this decides whether the render is worth spending.

resolve_dead(renderer, code, W, H) -> {"live": [...], "dead": [...], ...}
where each token is {idx, kind, value, start, end, css} and css is the set of
computed-style properties that moved when the token was ablated in the DOM.
"""
from __future__ import annotations

# One page.evaluate: loop every traced element, ablate each of its class
# utilities / inline-style decls / colour-ish attributes, and record which
# computed-style properties change on the element AND its descendants (a parent
# utility like `flex` reshapes its children, so element-only diffing misses it).
_JS = r"""
() => {
  const CAP = 12;                       // descendants snapshotted per element
  const snap = (nodes) => nodes.map(el => {
    const s = getComputedStyle(el), o = {};
    for (let i = 0; i < s.length; i++) o[s[i]] = s.getPropertyValue(s[i]);
    return o;
  });
  const diff = (before, after) => {
    const changed = new Set();
    for (let i = 0; i < before.length; i++) {
      const b = before[i], a = after[i];
      for (const p in b) if (b[p] !== a[p]) changed.add(p);
    }
    return [...changed];
  };
  const out = [];
  for (const el of document.querySelectorAll('[data-w2c-src]')) {
    const src = el.getAttribute('data-w2c-src');
    const nodes = [el, ...el.querySelectorAll('*')].slice(0, CAP);

    // --- class utilities ---
    const orig = el.getAttribute('class');
    if (orig !== null) {
      const cls = orig.split(/\s+/).filter(Boolean);
      for (let i = 0; i < cls.length; i++) {
        const before = snap(nodes);
        el.setAttribute('class', cls.filter((_, j) => j !== i).join(' '));
        const changed = diff(before, snap(nodes));
        el.setAttribute('class', orig);               // restore exact string, not classList.add
        out.push({src, kind: 'utility', value: cls[i], css: changed});
      }
    }

    // --- inline style declarations ---
    const st = el.getAttribute('style');
    if (st) {
      for (const decl of st.split(';').map(s => s.trim()).filter(Boolean)) {
        const prop = decl.split(':')[0].trim();
        const before = snap(nodes);
        el.style.removeProperty(prop);
        const changed = diff(before, snap(nodes));
        el.setAttribute('style', st);
        out.push({src, kind: 'style_decl', value: decl, css: changed});
      }
    }

    // --- colour-ish / geometry SVG + host attributes ---
    for (const name of ['fill','stroke','width','height','r','cx','cy','opacity','stroke-width']) {
      if (!el.hasAttribute(name)) continue;
      const v = el.getAttribute(name);
      const before = snap(nodes);
      el.removeAttribute(name);
      const changed = diff(before, snap(nodes));
      el.setAttribute(name, v);
      out.push({src, kind: 'attr:' + name, value: v, css: changed});
    }
  }
  return out;
}
"""


def resolve_dead(renderer, code: str, width: int, height: int, out_png=None):
    """Return live/dead classification for every DOM-ablatable token.

    A token is DEAD when ablating it changes no computed style property, so it
    cannot move the render and its ablation render can be skipped.
    """
    import tempfile
    from pathlib import Path
    png = Path(out_png) if out_png else Path(tempfile.mktemp(suffix=".png"))
    tr = renderer.render_with_trace(jsx=code, css="", width=width,
                                    height=height, out_path=png)
    # reuse the page the renderer just built; if the renderer tears it down,
    # fall back to a fresh context on the same browser.
    page = getattr(renderer, "_last_page", None)
    if page is None:
        recs = _eval_fresh(renderer, code, width, height)
    else:
        recs = page.evaluate(_JS)
    live = [r for r in recs if r["css"]]
    dead = [r for r in recs if not r["css"]]
    return {"live": live, "dead": dead, "n": len(recs),
            "trace": tr, "png": str(png)}


def _eval_fresh(renderer, code, width, height):
    """Rebuild the page once to run the resolver JS (renderer closed its page)."""
    import tempfile, os
    html = renderer._build_html(code, "", trace=True)
    ctx = renderer._new_context(width, height)
    page = ctx.new_page()
    tmp = tempfile.NamedTemporaryFile(suffix=".html", mode="w", delete=False, dir="/tmp")
    tmp.write(html); tmp.close()
    try:
        page.goto(f"file://{tmp.name}", wait_until="domcontentloaded")
        renderer._wait_until_stable(page, 15000)
        return page.evaluate(_JS)
    finally:
        ctx.close(); os.unlink(tmp.name)


def dead_value_map(resolved):
    SOUND = ("utility", "style_decl")
    dead = {}
    for r in resolved["dead"]:
        if r["kind"] in SOUND:
            dead.setdefault(r["src"], set()).add(r["value"])
    for r in resolved["live"]:
        if r["src"] in dead:
            dead[r["src"]].discard(r["value"])
    return dead
