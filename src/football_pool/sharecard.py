"""The link-preview card: the scoreboard, as an image.

The site's whole reason to exist is the family group chat, and a link in a
group chat is judged by its unfurl. A generic preview says "a website"; this
card says who is winning. It is generated at build time like every page, drawn
in the site's own palette, and served from ``/assets/`` so it carries a content
fingerprint — a changed leaderboard is a URL no scraper has cached.

Deliberately primitive inputs (strings and pairs, no DataFrames) so the
renderer is trivial to test and can never take the build down over a pandas
edge case. The caller decides what the card says; this module only draws it.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .season import REPO_ROOT

WIDTH, HEIGHT = 1200, 630

# The site's dark tokens, verbatim. The card is the site in miniature, and a
# preview in a different palette would read as somebody else's link.
FIELD = (7, 11, 14)
TURF_B = (10, 16, 20)
CHALK = (232, 237, 240)
CHALK_DIM = (143, 163, 174)
RULE = (36, 52, 62)
LIVE = (198, 255, 61)
MODEL = (77, 216, 230)

FONT_DIR = REPO_ROOT / "assets" / "fonts" / "sharecard"

# Mowing bands share the 40px yard-line rhythm the page background uses,
# scaled up so they read at preview size.
BAND = 60


def _font(name: str, size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """A card font, or Pillow's default if the file has gone missing.

    The default font is ugly, but an ugly card is strictly better than a build
    that cannot publish the leaderboard because a font file was deleted.
    """
    try:
        font = ImageFont.truetype(str(FONT_DIR / name), size)
        if bold:
            try:
                font.set_variation_by_axes([700])
            except OSError:
                pass
        return font
    except OSError:
        return ImageFont.load_default(size)


def draw_card(
    out_path: Path,
    *,
    year: int,
    headline: str,
    sub: str,
    lines: list[tuple[str, str, str]],
    modelled: bool = False,
) -> Path:
    """Draw the card and return its path.

    ``lines`` is up to five ``(lead, name, value)`` rows — rank/name/points in
    season, or name/win-chance before kickoff. ``modelled`` renders the values
    in the model's cyan so a forecast can never be mistaken for something that
    already happened, the same rule the site itself lives by.
    """
    img = Image.new("RGB", (WIDTH, HEIGHT), FIELD)
    d = ImageDraw.Draw(img)

    # The field: alternating mown bands with a hairline where they meet.
    for x in range(0, WIDTH, BAND * 2):
        d.rectangle([x + BAND, 0, x + BAND * 2, HEIGHT], fill=TURF_B)
    for x in range(BAND, WIDTH, BAND):
        d.line([(x, 0), (x, HEIGHT)], fill=(15, 22, 27), width=1)

    display = _font("big-shoulders.ttf", 84, bold=True)
    body = _font("instrument-sans.ttf", 30)
    mono = _font("plex-mono.ttf", 34)
    mono_small = _font("plex-mono.ttf", 24)

    margin = 70
    d.text((margin, 52), f"FAMILY POOL {year}", font=display, fill=LIVE)
    d.text((margin, 158), headline, font=body, fill=CHALK)
    d.text((margin, 202), sub, font=mono_small, fill=CHALK_DIM)
    if modelled:
        tag = "modelled"
        d.text(
            (WIDTH - margin - d.textlength(tag, font=mono_small), 202),
            tag,
            font=mono_small,
            fill=MODEL,
        )

    d.line([(margin, 258), (WIDTH - margin, 258)], fill=RULE, width=2)

    value_fill = MODEL if modelled else LIVE
    y = 292
    for lead, name, value in lines[:5]:
        d.text((margin, y), lead, font=mono, fill=CHALK_DIM)
        d.text((margin + 80, y), name, font=body, fill=CHALK)
        right = d.textlength(value, font=mono)
        d.text((WIDTH - margin - right, y), value, font=mono, fill=value_fill)
        y += 62

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, optimize=True)
    return out_path


def card_content(state: dict, rows: list[dict], forecast: dict | None) -> dict:
    """What the card should say right now, from what the pages already know.

    In season the answer is the leaderboard. Before kickoff every actual total
    is 0.00, so the model's favourites are the only interesting numbers — and
    when there is no model either, the pot and the field are the story.
    """
    pot = f"${state['pot']:.0f} pot"
    field = f"{state['entrants']} in"

    if state.get("week"):
        return {
            "headline": "The standings",
            "sub": f"{state['phase']} · {pot} · {field}",
            "lines": [
                (f"{r['rank']}", r["name"], f"{r['banked']:.2f}")
                for r in rows[:5]
            ],
            "modelled": False,
        }

    if forecast and forecast.get("finish_rows"):
        return {
            "headline": "Kickoff soon — the model's favourites",
            "sub": f"{pot} · {field} · winner takes the top share",
            "lines": [
                (f"{i + 1}", r["name"], f"{r['probs'][0] * 100:.0f}%")
                for i, r in enumerate(forecast["finish_rows"][:5])
            ],
            "modelled": True,
        }

    return {
        "headline": "Kickoff soon",
        "sub": f"{pot} · {field}",
        "lines": [(f"{i + 1}", r["name"], "0.00") for i, r in enumerate(rows[:5])],
        "modelled": False,
    }
