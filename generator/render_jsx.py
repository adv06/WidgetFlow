"""JSX + CSS renderer.

Wraps a single JSX component plus a CSS string into a self-contained HTML
page (React + Babel-standalone + Tailwind via CDN), opens it in headless
chromium, and screenshots the rendered widget.

Designed to be used as a context manager from one process; a single browser
is reused across calls.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import queue
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Optional

# Node.js binary for server-side JSX transpilation (Babel runs once per render,
# server-side, instead of inside each Chromium context — saves ~3s per render).
_NODE_BIN = "/root/.nvm/versions/node/v24.18.0/bin/node"

# Local copies of CDN assets so Chromium can render without network access.
_STATIC = Path(__file__).parent / "static"

from playwright.sync_api import sync_playwright, Browser, Playwright


_IMPORT_RE = re.compile(r"^\s*import\s+[^\n;]+;?\s*$", re.MULTILINE)
_EXPORT_DEFAULT_RE = re.compile(r"\bexport\s+default\s+")
_EXPORT_RE = re.compile(r"^\s*export\s+(?=(function|const|let|var)\b)", re.MULTILINE)
_TAG_NAME_RE = re.compile(r"[A-Za-z][A-Za-z0-9:.-]*")


def _make_browser_safe(jsx: str) -> str:
    """Strip ES module syntax that Babel-standalone with the `react` preset can't transform.
    The training-target convention is `export default function Widget()`; the browser
    harness expects a global `Widget` function, so we strip the module keywords here.
    """
    jsx = _IMPORT_RE.sub("", jsx)
    jsx = _EXPORT_DEFAULT_RE.sub("", jsx)
    jsx = _EXPORT_RE.sub("", jsx)
    return jsx


def _find_jsx_tag_end(src: str, start: int) -> int:
    quote: str | None = None
    brace_depth = 0
    i = start + 1
    while i < len(src):
        ch = src[i]
        prev = src[i - 1] if i > 0 else ""
        if quote is not None:
            if ch == quote and prev != "\\":
                quote = None
        elif ch in ("'", '"', "`"):
            quote = ch
        elif ch == "{":
            brace_depth += 1
        elif ch == "}":
            brace_depth = max(0, brace_depth - 1)
        elif ch == ">" and brace_depth == 0:
            return i
        i += 1
    return -1


def _instrument_jsx_source(jsx: str) -> str:
    """Attach raw-source spans to JSX host tags before Babel compiles them."""
    out: list[str] = []
    last = 0
    i = 0
    while i < len(jsx):
        if jsx[i] != "<" or i + 1 >= len(jsx) or jsx[i + 1] in "/!?":
            i += 1
            continue
        match = _TAG_NAME_RE.match(jsx, i + 1)
        if not match:
            i += 1
            continue
        name = match.group(0)
        # Only host elements produce DOM nodes. Component names are usually uppercase.
        if not name or not name[0].islower():
            i += 1
            continue
        end = _find_jsx_tag_end(jsx, i)
        if end < 0:
            i += 1
            continue
        tag_text = jsx[i : end + 1]
        if "data-w2c-src" not in tag_text:
            out.append(jsx[last : match.end()])
            out.append(f' data-w2c-src="{i}:{end + 1}"')
            last = match.end()
        i = end + 1
    if not out:
        return jsx
    out.append(jsx[last:])
    return "".join(out)


def _static_url(name: str) -> str:
    """Return a file:// URL for a bundled static asset."""
    p = _STATIC / name
    if p.exists():
        return f"file://{p.resolve()}"
    # fallback to CDN if local copy missing
    _CDN = {
        "react.production.min.js": "https://unpkg.com/react@18.3.1/umd/react.production.min.js",
        "react-dom.production.min.js": "https://unpkg.com/react-dom@18.3.1/umd/react-dom.production.min.js",
        "babel.min.js": "https://unpkg.com/@babel/standalone@7.24.7/babel.min.js",
        "tailwind.js": "https://cdn.tailwindcss.com",
    }
    return _CDN.get(name, name)


def _transpile_jsx(jsx: str) -> str | None:
    """Pre-transpile JSX → plain JS via Node.js + babel standalone.

    Runs once per render call, server-side, eliminating in-browser Babel
    compilation (~3s per render in Chromium). Returns None on any failure so
    callers can fall back to the browser-side Babel path.
    """
    babel_path = str(_STATIC / "babel.min.js")
    if not os.path.exists(_NODE_BIN) or not os.path.exists(babel_path):
        return None
    safe = _make_browser_safe(jsx)
    script = (
        f"const Babel=require({json.dumps(babel_path)});"
        f"try{{const r=Babel.transform({json.dumps(safe)},{{presets:['react']}});"
        f"process.stdout.write(r.code);}}catch(e){{process.exit(1);}}"
    )
    try:
        out = subprocess.run(
            [_NODE_BIN, "-e", script],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout
    except Exception:
        pass
    return None


# HTML template used when JSX is pre-compiled server-side (no Babel.js needed).
HTML_TEMPLATE_COMPILED = r"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <script src="__REACT_URL__"></script>
  <script src="__REACT_DOM_URL__"></script>
  <script src="__TAILWIND_URL__"></script>
  <style>
    html, body { margin: 0; padding: 0; background: #ffffff; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    #root { display: inline-block; }
__CUSTOM_CSS__
  </style>
</head>
<body>
  <div id="root"></div>
  <script>
__JS_SOURCE__
    const _root = ReactDOM.createRoot(document.getElementById('root'));
    _root.render(React.createElement(Widget));
    window.__widget_rendered__ = true;
  </script>
</body>
</html>
"""

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <script src="__REACT_URL__"></script>
  <script src="__REACT_DOM_URL__"></script>
  <script src="__BABEL_URL__"></script>
  <script src="__TAILWIND_URL__"></script>
  <style>
    html, body { margin: 0; padding: 0; background: #ffffff; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    #root { display: inline-block; }
__CUSTOM_CSS__
  </style>
</head>
<body>
  <div id="root"></div>
  <script type="text/babel" data-presets="react">
__JSX_SOURCE__
    const _root = ReactDOM.createRoot(document.getElementById('root'));
    _root.render(React.createElement(Widget));
    window.__widget_rendered__ = true;
  </script>
</body>
</html>
"""


class JSXRenderer:
    def __init__(self) -> None:
        self._pw: Optional[Playwright] = None
        self._browser: Optional[Browser] = None

    def __enter__(self) -> "JSXRenderer":
        self._pw = sync_playwright().start()
        extra_args = shlex.split(os.environ.get("W2C_CHROMIUM_ARGS", ""))
        executable = os.environ.get("W2C_CHROMIUM_EXECUTABLE")
        launch_kwargs = {}
        if executable:
            if not Path(executable).is_file():
                raise FileNotFoundError(f"W2C Chromium executable not found: {executable}")
            launch_kwargs["executable_path"] = executable
        self._browser = self._pw.chromium.launch(
            headless=True,
            chromium_sandbox=False,
            args=[
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-seccomp-filter-sandbox",
                "--disable-gpu-sandbox",
                "--disable-gpu",
                "--no-zygote",
                *extra_args,
            ],
            **launch_kwargs,
        )
        return self

    def __exit__(self, *exc) -> None:
        try:
            if self._browser is not None:
                self._browser.close()
        finally:
            if self._pw is not None:
                self._pw.stop()

    def _build_html(self, jsx: str, css: str, trace: bool = False) -> str:
        source = _instrument_jsx_source(jsx) if trace else jsx
        compiled = _transpile_jsx(source)
        if compiled is not None:
            return (
                HTML_TEMPLATE_COMPILED
                .replace("__REACT_URL__", _static_url("react.production.min.js"))
                .replace("__REACT_DOM_URL__", _static_url("react-dom.production.min.js"))
                .replace("__TAILWIND_URL__", _static_url("tailwind.js"))
                .replace("__CUSTOM_CSS__", css or "")
                .replace("__JS_SOURCE__", compiled)
            )
        # Fallback: browser-side Babel (network-free via local file://)
        return (
            HTML_TEMPLATE
            .replace("__REACT_URL__", _static_url("react.production.min.js"))
            .replace("__REACT_DOM_URL__", _static_url("react-dom.production.min.js"))
            .replace("__BABEL_URL__", _static_url("babel.min.js"))
            .replace("__TAILWIND_URL__", _static_url("tailwind.js"))
            .replace("__CUSTOM_CSS__", css or "")
            .replace("__JSX_SOURCE__", _make_browser_safe(source))
        )

    def _new_context(self, width: int, height: int):
        ctx = self._browser.new_context(
            viewport={"width": max(width + 40, 400), "height": max(height + 40, 400)},
            device_scale_factor=2,
            locale="en-US",
            color_scheme="light",
            reduced_motion="reduce",
        )
        # All render assets are bundled locally (file://). Abort any external
        # http(s) request (e.g. webfonts) so a slow/blocked fetch can't stall the
        # render — an unresolved external font was a source of indefinite hangs.
        ctx.route(
            "**/*",
            lambda route: route.abort()
            if route.request.url.startswith(("http://", "https://"))
            else route.continue_(),
        )
        return ctx

    def _wait_until_stable(self, page, timeout_ms: int) -> None:
        page.wait_for_function(
            "() => window.__widget_rendered__ === true",
            timeout=timeout_ms,
        )
        page.add_style_tag(content="""
          *, *::before, *::after {
            animation-duration: 0s !important;
            animation-delay: 0s !important;
            transition-duration: 0s !important;
            transition-delay: 0s !important;
            caret-color: transparent !important;
          }
        """)
        try:
            # Cap the font wait: document.fonts.ready never resolves if a webfont
            # request hangs, and page.evaluate has no timeout of its own.
            page.evaluate("""() => Promise.race([
                (document.fonts && document.fonts.ready) ? document.fonts.ready : Promise.resolve(),
                new Promise((res) => setTimeout(res, 200)),
            ])""")
        except Exception:
            pass
        page.wait_for_timeout(int(os.environ.get("W2C_RENDER_SETTLE_MS", "100")))

    # Screenshots come back at device pixels (device_scale_factor=2 in
    # _new_context) — kept at 2 for render quality.
    _DEVICE_SCALE = 2

    def _screenshot_element(self, element, out_path: Path,
                            width: int | None = None, height: int | None = None) -> None:
        try:
            element.screenshot(path=str(out_path), omit_background=False, animations="disabled")
        except TypeError:
            element.screenshot(path=str(out_path), omit_background=False)
        if width and height:
            self._normalize_to_size(out_path, int(width), int(height), self._DEVICE_SCALE)

    @staticmethod
    def _normalize_to_size(out_path: Path, width: int, height: int, device_scale: int) -> None:
        """Emit the PNG at exactly (width, height) CSS pixels via scale-to-fit.

        One resize handles both effects: the device_scale x screenshot comes
        down to CSS pixels (the 2x dims alone were tanking the bench geometry
        score to ~0.61 on every render), and any overflow is scaled into the
        target canvas rather than cropped away. Scale-to-fit is what the
        pipeline has always effectively scored (resize_to_match squeezed the
        full render into GT dims) and matches both the real bench CLI and
        widget2code's own auto-resize; cropping instead was tried and tanked
        layout/style/perceptual by 15-20 points.

        render_with_trace bbox coords are CSS pixels; for exact-fit renders
        (the common case) they stay aligned 1:1 with the output."""
        from PIL import Image as _PILImage
        img = _PILImage.open(out_path).convert("RGB")
        if img.size != (width, height):
            img = img.resize((width, height), _PILImage.LANCZOS)
        img.save(out_path)

    def render(
        self,
        jsx: str,
        css: str,
        width: int,
        height: int,
        out_path: Path,
        timeout_ms: int = 15000,
    ) -> None:
        if self._browser is None:
            raise RuntimeError("JSXRenderer must be used as a context manager")
        html = self._build_html(jsx, css)
        context = self._new_context(width, height)
        page = context.new_page()
        tmp = tempfile.NamedTemporaryFile(suffix=".html", mode="w", delete=False, dir="/tmp")
        try:
            tmp.write(html)
            tmp.flush()
            tmp.close()
            page.goto(f"file://{tmp.name}", wait_until="domcontentloaded", timeout=timeout_ms)
            self._wait_until_stable(page, timeout_ms)
            element = page.locator("#root").first
            self._screenshot_element(element, out_path, width, height)
        finally:
            context.close()
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

    def render_with_trace(
        self,
        jsx: str,
        css: str,
        width: int,
        height: int,
        out_path: Path,
        timeout_ms: int = 15000,
    ) -> dict:
        """Render and return DOM boxes mapped to raw JSX source spans."""
        if self._browser is None:
            raise RuntimeError("JSXRenderer must be used as a context manager")
        html = self._build_html(jsx, css, trace=True)
        context = self._new_context(width, height)
        page = context.new_page()
        tmp = tempfile.NamedTemporaryFile(suffix=".html", mode="w", delete=False, dir="/tmp")
        try:
            tmp.write(html)
            tmp.flush()
            tmp.close()
            page.goto(f"file://{tmp.name}", wait_until="domcontentloaded", timeout=timeout_ms)
            self._wait_until_stable(page, timeout_ms)
            element = page.locator("#root").first
            self._screenshot_element(element, out_path, width, height)
            return page.evaluate(
                """() => {
                  const root = document.getElementById('root');
                  const rr = root.getBoundingClientRect();
                  const elems = Array.from(document.querySelectorAll('[data-w2c-src]'));
                  // index each traced node so we can report true DOM nesting.
                  // Box containment cannot recover it: a wrapper whose child
                  // exactly fills it is indistinguishable from a leaf.
                  const _ix = new Map();
                  elems.forEach((el, i) => _ix.set(el, i));
                  const _parentOf = (el) => {
                    let p = el.parentElement;
                    while (p) { if (_ix.has(p)) return _ix.get(p); p = p.parentElement; }
                    return -1;
                  };
                  return {
                    root_bbox: [0, 0, rr.width, rr.height],
                    elements: elems.map((el, idx) => {
                      const r = el.getBoundingClientRect();
                      const st = window.getComputedStyle(el);
                      const src = el.getAttribute('data-w2c-src') || '';
                      const cls = typeof el.className === 'string' ? el.className : '';
                      return {
                        idx,
                        parent: _parentOf(el),
                        tag: el.tagName.toLowerCase(),
                        src,
                        bbox: [r.left - rr.left, r.top - rr.top, r.right - rr.left, r.bottom - rr.top],
                        display: st.display,
                        visibility: st.visibility,
                        opacity: st.opacity,
                        className: cls.slice(0, 240),
                        text: (el.innerText || el.textContent || '').slice(0, 120)
                      };
                    })
                  };
                }"""
            )
        finally:
            context.close()
            try:
                os.unlink(tmp.name)
            except OSError:
                pass


    def render_batch(
        self,
        tasks: list[tuple[str, str, int, int, "Path", int]],
        n_workers: int = 4,
        retries: int = 1,
    ) -> list[Exception | None]:
        """Render multiple JSXes in parallel. Returns a list of errors (None = success).

        Each task is (jsx, css, width, height, out_path, timeout_ms).
        Playwright's sync API is not thread-safe (its asyncio event loop is pinned to
        the creating thread), so each worker thread owns one JSXRenderer/browser for
        its whole lifetime — created, used, and closed on that same thread — and reuses
        it across every task it drains. (Each render() still opens a fresh browser
        context, so a bad render can't poison later ones.) This avoids cold-launching a
        Chromium process per render, which dominated batch wall-clock.
        """
        if not tasks:
            return []
        results: list[Exception | None] = [None] * len(tasks)
        task_q: "queue.Queue[int]" = queue.Queue()
        for i in range(len(tasks)):
            task_q.put(i)

        def _drain() -> None:
            with JSXRenderer() as r:
                while True:
                    try:
                        idx = task_q.get_nowait()
                    except queue.Empty:
                        return
                    jsx, css, w, h, out_path, timeout_ms = tasks[idx]
                    last_err: Exception | None = None
                    for _ in range(max(1, retries)):
                        try:
                            r.render(jsx, css, w, h, out_path, timeout_ms)
                            last_err = None
                            break
                        except Exception as e:
                            last_err = e
                    results[idx] = last_err

        threads = [
            threading.Thread(target=_drain, daemon=True)
            for _ in range(min(n_workers, len(tasks)))
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return results


def render_once(jsx: str, css: str, width: int, height: int, out_path: Path) -> None:
    """Convenience helper for one-off renders. For batch use the context manager."""
    with JSXRenderer() as r:
        r.render(jsx, css, width, height, out_path)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("usage: render_jsx.py <input.json> <out.png>")
        sys.exit(2)
    inp = json.loads(Path(sys.argv[1]).read_text())
    out = Path(sys.argv[2])
    render_once(inp["jsx"], inp.get("css", ""), inp["width"], inp["height"], out)
    print(f"wrote {out}")
