"""Convert a meeting markdown file (this dir's format) to a .pptx deck.

Slide schema we expect inside the markdown:
  # <Deck title>            -> title slide
  optional **Presenter:** / **Slides:** / reference list before first '---'
  ## Slide N — <Title>      -> one content slide per occurrence
    - bullet
    - bullet (supports **bold** -> bold runs)
  ## <Other heading>        -> also rendered as a content slide
  --- separators are ignored

Usage:
    uv run --with python-pptx python md_to_pptx.py <input.md> <output.pptx>
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt


BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


def parse_md(md_path: Path):
    text = md_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    deck_title = ""
    intro_lines: list[str] = []
    slides: list[tuple[str, list[str]]] = []
    cur_title: str | None = None
    cur_bullets: list[str] = []
    seen_first_section = False

    def flush():
        nonlocal cur_title, cur_bullets
        if cur_title is not None:
            slides.append((cur_title, cur_bullets))
        cur_title, cur_bullets = None, []

    for raw in lines:
        line = raw.rstrip()
        if not deck_title and line.startswith("# "):
            deck_title = line[2:].strip()
            continue
        if line.startswith("## "):
            flush()
            cur_title = line[3:].strip()
            seen_first_section = True
            continue
        if line.strip() == "---":
            continue
        if cur_title is None:
            if seen_first_section:
                continue
            if line.strip():
                intro_lines.append(line)
            continue
        m = re.match(r"\s*[-*]\s+(.*)", line)
        if m:
            cur_bullets.append(m.group(1).strip())
            continue
        # Continuation line: append to last bullet if present
        if line.strip() and cur_bullets:
            cur_bullets[-1] += " " + line.strip()
    flush()
    return deck_title, intro_lines, slides


def add_runs(paragraph, text: str):
    # Strip markdown links to "label" (we lose URL in pptx body — title slide gets links separately)
    text = LINK_RE.sub(r"\1", text)
    parts = []
    last = 0
    for m in BOLD_RE.finditer(text):
        if m.start() > last:
            parts.append((text[last : m.start()], False))
        parts.append((m.group(1), True))
        last = m.end()
    if last < len(text):
        parts.append((text[last:], False))
    if not parts:
        parts = [(text, False)]

    first = True
    for chunk, bold in parts:
        run = paragraph.add_run() if not first else paragraph.runs[0] if paragraph.runs else paragraph.add_run()
        run.text = chunk
        run.font.bold = bold
        run.font.size = Pt(18)
        first = False


def build_pptx(deck_title: str, intro_lines: list[str], slides, out_path: Path):
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    # Title slide
    title_layout = prs.slide_layouts[0]
    s = prs.slides.add_slide(title_layout)
    s.shapes.title.text = deck_title or out_path.stem
    if len(s.placeholders) > 1:
        sub = s.placeholders[1]
        sub.text = "\n".join(l for l in intro_lines if l.strip())

    # Content slides
    bullet_layout = prs.slide_layouts[1]
    for title, bullets in slides:
        s = prs.slides.add_slide(bullet_layout)
        s.shapes.title.text = title
        body = s.placeholders[1].text_frame
        body.clear()
        for i, b in enumerate(bullets):
            p = body.paragraphs[0] if i == 0 else body.add_paragraph()
            p.level = 0
            # populate via runs to support bold
            # ensure paragraph starts empty
            p.text = ""
            add_runs(p, b)

    prs.save(out_path)


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    in_md = Path(sys.argv[1]).resolve()
    out_pptx = Path(sys.argv[2]).resolve()
    deck_title, intro, slides = parse_md(in_md)
    build_pptx(deck_title, intro, slides, out_pptx)
    print(f"Wrote {out_pptx} ({len(slides)} content slides)")


if __name__ == "__main__":
    main()
