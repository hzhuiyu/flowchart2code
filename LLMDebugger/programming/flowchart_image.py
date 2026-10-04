"""Resolve and attach flowchart images for the image-LDB variant.

Enabled by env LDB_USE_IMAGE=1. Existing text-only LDB is unchanged when
the flag is off. Image files live at:

  data/<Dataset>/images/<orig_task_id>.png

where LDB task_id is `<Dataset>/<orig_task_id>` (e.g. HumanEval-V/HumanEval-0).
"""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Optional, Union

_THIS_DIR = Path(__file__).resolve().parent
_F2C_ROOT = _THIS_DIR.parent.parent

# Skip absurdly large payloads (flowchart PNGs here are typically 100–700 KB).
_MAX_IMAGE_BYTES = 4 * 1024 * 1024

_KNOWN_PREFIXES = ("HumanEval-V", "Algorithm", "MATH", "LiveCodeBench")


def use_image_enabled() -> bool:
    return os.getenv("LDB_USE_IMAGE", "").strip().lower() in ("1", "true", "yes", "on")


def message_text(content) -> str:
    """Plain text of a chat message, whether content is a str or multimodal list."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text") or "")
            elif isinstance(part, str):
                parts.append(part)
        return "\n".join(parts)
    return "" if content is None else str(content)


def has_image(content) -> bool:
    if not isinstance(content, list):
        return False
    return any(isinstance(p, dict) and p.get("type") == "image_url" for p in content)


def strip_images_from_messages(messages):
    """Drop image_url parts so later LDB iterations stay text-only.

    The first debug conversation keeps the flowchart; once messages already
    exist (cur_iter > 0), convert any multimodal content back to plain text.
    """
    if not messages or isinstance(messages, str):
        return messages
    for msg in messages:
        content = getattr(msg, "content", None)
        if has_image(content):
            msg.content = message_text(content)
    return messages


def resolve_flowchart_image(task_id: str) -> Optional[str]:
    """Map LDB task_id `Dataset/orig` -> data/Dataset/images/orig.{png,jpg,jpeg}."""
    if not task_id or "/" not in task_id:
        return None
    prefix, orig_id = task_id.split("/", 1)
    if prefix not in _KNOWN_PREFIXES or not orig_id:
        return None
    images_dir = _F2C_ROOT / "data" / prefix / "images"
    for ext in (".png", ".jpg", ".jpeg"):
        path = images_dir / f"{orig_id}{ext}"
        if path.is_file():
            return str(path)
    return None


def user_content_with_image(text: str, image_path: Optional[str]) -> Union[str, list]:
    """OpenAI chat `content`: str, or [text, image_url] when a flowchart is attached."""
    if not image_path or not use_image_enabled():
        return text
    path = Path(image_path)
    if not path.is_file():
        print(f"[image] missing file, fallback to text: {image_path}")
        return text
    try:
        size = path.stat().st_size
    except OSError as e:
        print(f"[image] stat failed ({e}), fallback to text: {image_path}")
        return text
    if size <= 0 or size > _MAX_IMAGE_BYTES:
        print(f"[image] skip (size={size}), fallback to text: {image_path}")
        return text
    suffix = path.suffix.lower()
    mime = "image/jpeg" if suffix in (".jpg", ".jpeg") else "image/png"
    try:
        b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    except OSError as e:
        print(f"[image] read failed ({e}), fallback to text: {image_path}")
        return text
    caption = (
        "A flowchart of the algorithm is attached. "
        "Use both the text description and the flowchart when writing the code.\n\n"
    )
    print(f"[image] attached {path.name} ({size} bytes)")
    return [
        {"type": "text", "text": caption + text},
        {
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{b64}", "detail": "high"},
        },
    ]
