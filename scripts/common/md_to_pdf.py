"""Render one of the project's Markdown documents as a PDF.

Written because the documents here are the thesis's working notes and get read
away from a terminal. It handles the constructs those documents actually use --
headings, paragraphs, fenced code, tables, bullet lists, and inline bold and
code -- rather than trying to be a general Markdown engine.

Tables are the reason this is not a two-line pandoc call: the ablation tables
carry the argument, and they have to survive onto the page with their columns
aligned and their ragged rows intact.

Usage::

    python scripts/common/md_to_pdf.py docs/ablation_ladder.md
    python scripts/common/md_to_pdf.py docs/ablation_ladder.md --out /tmp/ladder.pdf
"""

import argparse
import html
import pathlib
import re

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    ListFlowable,
    ListItem,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

_BOLD = re.compile(r"\*\*(.+?)\*\*")
_ITALIC = re.compile(r"(?<![*\w])\*([^*\n]+?)\*(?![*\w])")
_CODE = re.compile(r"`([^`]+?)`")
_LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_ALIGNMENT_ROW = re.compile(r"^\|[\s:|-]+\|$")


def inline(text: str) -> str:
    """Convert one line of inline Markdown to reportlab's mini-HTML.

    Escaping happens first, so a literal ``<`` in the source cannot become
    markup, and the tags inserted afterwards are the only ones reportlab sees.

    Args:
        text: A line of Markdown.

    Returns:
        The line as reportlab paragraph markup.
    """
    text = html.escape(text, quote=False)
    text = _LINK.sub(r'<link href="\2" color="#1a4f8a">\1</link>', text)
    text = _BOLD.sub(r"<b>\1</b>", text)
    text = _ITALIC.sub(r"<i>\1</i>", text)
    return _CODE.sub(
        r'<font face="Courier" size="9" backColor="#f2f2f2">\1</font>',
        text,
    )


def split_row(line: str) -> list[str]:
    """Split one Markdown table row into its cells.

    Args:
        line: A line beginning and ending with a pipe.

    Returns:
        The cell contents, unstripped of Markdown.
    """
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def parse_alignment(line: str) -> list[int]:
    """Read a Markdown alignment row into reportlab alignment constants.

    Args:
        line: The ``| :--- | ---: |`` row under a table header.

    Returns:
        One alignment per column.
    """
    alignments = []
    for cell in split_row(line):
        left, right = cell.startswith(":"), cell.endswith(":")
        if left and right:
            alignments.append(TA_CENTER)
        elif right:
            alignments.append(TA_RIGHT)
        else:
            alignments.append(TA_LEFT)
    return alignments


def build_table(
    rows: list[list[str]],
    styles: dict,
    width: float,
    alignments: list[int] | None = None,
) -> Table:
    """Lay out a Markdown table, padding ragged rows rather than dropping them.

    Args:
        rows: Header row first, then the body rows.
        styles: The paragraph styles to draw cells with.
        width: Available frame width.
        alignments: Per-column alignment; numeric columns read far better
            right-aligned, and these tables are mostly numbers.

    Returns:
        The flowable table.
    """
    columns = max(len(row) for row in rows)
    padded = [row + [""] * (columns - len(row)) for row in rows]

    # Give each column room in proportion to its longest cell, but never let one
    # column starve the others.
    longest = [
        max(len(padded[r][c]) for r in range(len(padded))) for c in range(columns)
    ]
    floor = 0.6 * sum(longest) / columns
    weights = [max(value, floor) for value in longest]
    total = sum(weights)
    widths = [width * weight / total for weight in weights]

    per_column = {}
    for column in range(columns):
        alignment = (alignments or [])[column] if alignments and column < len(alignments) else TA_LEFT
        for key in ("th", "td"):
            per_column[(key, column)] = ParagraphStyle(
                f"{key}{column}",
                parent=styles[key],
                alignment=alignment,
            )
    data = [
        [
            Paragraph(
                inline(cell),
                per_column[("th" if index == 0 else "td", column)],
            )
            for column, cell in enumerate(row)
        ]
        for index, row in enumerate(padded)
    ]
    table = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef2f6")),
                ("LINEBELOW", (0, 0), (-1, 0), 0.9, colors.HexColor("#9fb3c8")),
                ("LINEBELOW", (0, 1), (-1, -2), 0.25, colors.HexColor("#dfe6ec")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ],
        ),
    )
    return table


def make_styles() -> dict:
    """The paragraph styles the renderer draws with.

    Returns:
        A mapping of style name to style.
    """
    base = getSampleStyleSheet()
    body = ParagraphStyle(
        "body",
        parent=base["BodyText"],
        fontSize=9.7,
        leading=14.2,
        spaceAfter=7,
        alignment=TA_LEFT,
    )
    return {
        "body": body,
        "h1": ParagraphStyle(
            "h1",
            parent=base["Title"],
            fontSize=19,
            leading=24,
            spaceAfter=14,
            alignment=TA_LEFT,
            textColor=colors.HexColor("#12314f"),
        ),
        "h2": ParagraphStyle(
            "h2",
            parent=base["Heading2"],
            fontSize=13,
            leading=17,
            spaceBefore=16,
            spaceAfter=7,
            textColor=colors.HexColor("#1a4f8a"),
        ),
        "h3": ParagraphStyle(
            "h3",
            parent=base["Heading3"],
            fontSize=11,
            leading=15,
            spaceBefore=11,
            spaceAfter=5,
            textColor=colors.HexColor("#2c3e50"),
        ),
        "code": ParagraphStyle(
            "code",
            parent=body,
            fontName="Courier",
            fontSize=8.2,
            leading=10.6,
            leftIndent=7,
            spaceAfter=9,
            textColor=colors.HexColor("#1f2933"),
        ),
        "th": ParagraphStyle(
            "th",
            parent=body,
            fontName="Helvetica-Bold",
            fontSize=8.6,
            leading=11.4,
            spaceAfter=0,
        ),
        "td": ParagraphStyle(
            "td",
            parent=body,
            fontSize=8.6,
            leading=11.4,
            spaceAfter=0,
        ),
        "li": ParagraphStyle("li", parent=body, spaceAfter=3),
    }


def render(source: str, styles: dict, width: float) -> list:
    """Turn Markdown source into a list of flowables.

    Args:
        source: The document text.
        styles: Paragraph styles.
        width: Available frame width, for sizing tables.

    Returns:
        The flowables, in order.
    """
    story: list = []
    paragraph: list[str] = []
    code: list[str] | None = None
    table: list[list[str]] | None = None
    alignments: list[int] | None = None
    bullets: list[str] = []

    def flush_paragraph() -> None:
        if paragraph:
            story.append(Paragraph(inline(" ".join(paragraph)), styles["body"]))
            paragraph.clear()

    def flush_bullets() -> None:
        if bullets:
            story.append(
                ListFlowable(
                    [
                        ListItem(Paragraph(inline(item), styles["li"]), leftIndent=13)
                        for item in bullets
                    ],
                    bulletType="bullet",
                    bulletFontSize=6,
                    leftIndent=13,
                    spaceAfter=7,
                ),
            )
            bullets.clear()

    def flush_table() -> None:
        nonlocal table, alignments
        if table:
            # Keeping a table with the line above it stops a heading stranding
            # itself at the foot of a page.
            story.append(build_table(table, styles, width, alignments))
            story.append(Spacer(1, 9))
            table = None
            alignments = None

    for raw in source.splitlines():
        line = raw.rstrip()

        if line.startswith("```"):
            if code is None:
                flush_paragraph()
                flush_bullets()
                flush_table()
                code = []
            else:
                story.append(
                    Paragraph(
                        "<br/>".join(
                            html.escape(c, quote=False).replace(" ", "&nbsp;")
                            for c in code
                        ),
                        styles["code"],
                    ),
                )
                code = None
            continue
        if code is not None:
            code.append(line)
            continue

        if line.startswith("|") and line.endswith("|"):
            if _ALIGNMENT_ROW.match(line):
                alignments = parse_alignment(line)
                continue
            flush_paragraph()
            flush_bullets()
            if table is None:
                table = []
            table.append(split_row(line))
            continue
        flush_table()

        if not line.strip():
            flush_paragraph()
            flush_bullets()
            continue

        heading = re.match(r"^(#{1,3})\s+(.*)$", line)
        if heading:
            flush_paragraph()
            flush_bullets()
            level = len(heading.group(1))
            story.append(
                Paragraph(inline(heading.group(2)), styles[f"h{level}"]),
            )
            continue

        bullet = re.match(r"^[-*]\s+(.*)$", line)
        if bullet:
            flush_paragraph()
            bullets.append(bullet.group(1))
            continue

        if bullets:
            # A wrapped continuation of the bullet above it.
            bullets[-1] += " " + line.strip()
            continue

        paragraph.append(line.strip())

    flush_paragraph()
    flush_bullets()
    flush_table()
    return story


def main() -> None:
    """Render the document named on the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=pathlib.Path)
    parser.add_argument("--out", type=pathlib.Path, default=None)
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    out = args.out or args.source.with_suffix(".pdf")
    margin = 18 * mm
    width = A4[0] - 2 * margin

    document = SimpleDocTemplate(
        str(out),
        pagesize=A4,
        leftMargin=margin,
        rightMargin=margin,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title=args.title or args.source.stem.replace("_", " "),
    )
    styles = make_styles()
    story = render(args.source.read_text(encoding="utf-8"), styles, width)

    def footer(canvas, doc) -> None:
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.HexColor("#8b98a5"))
        canvas.drawString(margin, 9 * mm, args.source.name)
        canvas.drawRightString(A4[0] - margin, 9 * mm, str(doc.page))
        canvas.restoreState()

    document.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
