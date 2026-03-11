from __future__ import annotations

import base64
import io
import logging
from typing import Optional

from PIL import Image, ImageDraw, ImageFont

from agent.config import get_settings
from agent.models import MarkedElement

logger = logging.getLogger(__name__)
cfg = get_settings()

# Colour palette for SoM labels (cycles if >len)
_PALETTE = [
    "#FF3B30", "#FF9500", "#FFCC00", "#34C759", "#00C7BE",
    "#30B0C7", "#007AFF", "#5856D6", "#AF52DE", "#FF2D55",
]


def mark_screenshot(
    screenshot_b64: str,
    elements: list[MarkedElement],
    alpha: int = 200,
) -> str:
    """
    Overlay numbered bounding-box labels on a screenshot.

    Args:
        screenshot_b64: Base64-encoded PNG screenshot.
        elements: Interactable elements with bounding boxes.
        alpha: Opacity of label background (0-255).

    Returns:
        Base64-encoded marked PNG.
    """
    img_bytes = base64.b64decode(screenshot_b64)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                                   cfg.som_label_font_size)
    except OSError:
        font = ImageFont.load_default()

    for elem in elements[:cfg.som_max_elements]:
        color_hex = _PALETTE[elem.id % len(_PALETTE)]
        color_rgb = _hex_to_rgb(color_hex)

        x1, y1 = elem.x, elem.y
        x2, y2 = x1 + elem.width, y1 + elem.height

        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline=(*color_rgb, 200), width=2)

        # Draw label badge
        label = str(elem.id)
        bbox = draw.textbbox((0, 0), label, font=font)
        lw, lh = bbox[2] - bbox[0], bbox[3] - bbox[1]
        pad = 3
        badge_x2 = x1 + lw + pad * 2
        badge_y2 = y1 + lh + pad * 2

        draw.rectangle([x1, y1, badge_x2, badge_y2],
                        fill=(*color_rgb, alpha))
        draw.text((x1 + pad, y1 + pad), label, fill=(255, 255, 255, 255),
                   font=font)

    merged = Image.alpha_composite(img, overlay).convert("RGB")
    buf = io.BytesIO()
    merged.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))  # type: ignore
