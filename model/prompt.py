"""The prompt. Single source of truth.

Training, generation, probing and evaluation must all build messages the exact
same way. If any one of them drifts, the model is evaluated off-distribution
from how it was trained and the numbers are quietly wrong — no error, just a
worse score. Every call site imports from here; nothing constructs its own.
"""

from __future__ import annotations

from typing import Any

SYSTEM = (
    "You are an expert front-end engineer. Reproduce the widget in the image "
    "as a single self-contained React component using Tailwind CSS classes."
)

PROMPT = (
    "Write JSX for this widget. Return ONLY code, no explanation.\n"
    "Format: export default function Widget() { return (...); }"
)


def build_messages(image: Any, extra: str = "") -> tuple[list[dict], list[Any]]:
    """Chat messages + image list for one widget.

    `extra` appends to the user turn (reference hints, etc). It must be empty
    at eval time unless the model was trained with the same block, since a
    prompt the policy never saw in training degrades it.

    Returns (messages, images) — pass images separately to the processor.
    """
    user = PROMPT + (extra or "")
    msgs = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user}]},
    ]
    return msgs, [image]
