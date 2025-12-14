# src/utils/pretty_table.py
# -*- coding: utf-8 -*-
"""
Pretty Unicode table printer with optional per-column autofit.

Features:
- Autofit column widths: based on the widest formatted cell (header+rows) + symmetric padding.
- Optional fixed width (single int for all columns) or per-column widths (list[int]).
- Configurable float formatting and alignment.
- Optional title above the table.
- Returns the table string and also prints it (configurable).

No external dependencies.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Union

Number = Union[int, float]

__all__ = ["draw_table", "format_table"]


def draw_table(
        headers: Sequence[Any],
        rows: Sequence[Sequence[Any]],
        col_width: Optional[int] = None,
        col_widths: Optional[Sequence[int]] = None,
        padding: int = 1,
        float_fmt: str = ".3e",
        header_align: str = "center",
        cell_align: str = "center",
        title: Optional[str] = None,
        print_table: bool = True,
) -> str:
    """
    Build and print a Unicode box-drawn table.

    Args:
        headers: Column headers (length defines number of columns).
        rows: 2D data (each row must match the number of headers).
        col_width: Fixed width for all columns (overrides autofit). If None → autofit.
        col_widths: Optional per-column widths list. Overrides `col_width` and autofit if provided.
        padding: Spaces on each side of cell content when computing/printing widths.
        float_fmt: Format for float values (e.g., '.3e', '.4f').
        header_align: One of {'left','center','right'} for header text.
        cell_align: One of {'left','center','right'} for cell text.
        title: Optional title printed above the table.
        print_table: If True prints to stdout; in all cases returns the table string.

    Returns:
        The rendered table as a string.
    """
    table = format_table(
        headers=headers,
        rows=rows,
        col_width=col_width,
        col_widths=col_widths,
        padding=padding,
        float_fmt=float_fmt,
        header_align=header_align,
        cell_align=cell_align,
        title=title,
    )
    if print_table:
        print(table)
    return table


def format_table(
        headers: Sequence[Any],
        rows: Sequence[Sequence[Any]],
        col_width: Optional[int] = None,
        col_widths: Optional[Sequence[int]] = None,
        padding: int = 1,
        float_fmt: str = ".3e",
        header_align: str = "center",
        cell_align: str = "center",
        title: Optional[str] = None,
) -> str:
    """
    Same parameters as draw_table(), but only returns the table string (no print).
    """
    _validate_inputs(headers, rows)
    n_cols = len(headers)

    # 1) Format cells to strings for width measurement
    fmt_header_cells = [_format_value(h, float_fmt) for h in headers]
    fmt_rows: List[List[str]] = [[_format_value(v, float_fmt) for v in r] for r in rows]

    # 2) Determine column widths
    if col_widths is not None:
        if len(col_widths) != n_cols:
            raise ValueError(f"col_widths length ({len(col_widths)}) must equal number of columns ({n_cols}).")
        widths = list(col_widths)
    elif col_width is not None:
        widths = [int(col_width)] * n_cols
    else:
        # Autofit per column: max(len(cell)) across header+rows, then + 2*padding
        widths = []
        for j in range(n_cols):
            content_lengths = [len(fmt_header_cells[j])] + [len(fmt_rows[i][j]) for i in
                                                            range(len(fmt_rows))] if fmt_rows else [
                len(fmt_header_cells[j])]
            inner = max(content_lengths)  # without padding
            widths.append(inner + 2 * padding)

    # 3) Build border glyphs (heavy)
    top_left, top_mid, top_right = "┏", "┳", "┓"
    mid_left, mid_mid, mid_right = "┣", "╇", "┫"
    bot_left, bot_mid, bot_right = "┗", "┻", "┛"
    vbar = "┃"

    top_border = top_left + top_mid.join("━" * w for w in widths) + top_right
    sep_border = mid_left + mid_mid.join("━" * w for w in widths) + mid_right
    bot_border = bot_left + bot_mid.join("━" * w for w in widths) + bot_right

    # 4) Align and pad cells to target widths
    header_line = _render_row(fmt_header_cells, widths, padding, _align=header_align, vbar=vbar)
    row_lines = [
        _render_row(r, widths, padding, _align=cell_align, vbar=vbar)
        for r in fmt_rows
    ]

    # 5) Optional title (center to total width)
    lines: List[str] = []
    if title:
        total_inner = sum(widths) + (len(widths) - 1)  # account for vertical bars between columns
        # For a clean title line, align inside same visual width as table interior
        lines.append(title.center(total_inner + 2))  # +2 to account for outer bars spacing

    # 6) Stitch
    lines.append(top_border)
    lines.append(header_line)
    lines.append(sep_border)
    lines.extend(row_lines if row_lines else [])
    lines.append(bot_border)

    return "\n".join(lines)


# ---------- helpers ----------

def _validate_inputs(headers: Sequence[Any], rows: Sequence[Sequence[Any]]) -> None:
    n_cols = len(headers)
    for idx, r in enumerate(rows):
        if len(r) != n_cols:
            raise ValueError(f"Row {idx} length ({len(r)}) does not match headers length ({n_cols}).")


def _format_value(val: Any, float_fmt: str) -> str:
    # Floats → format; Ints → plain; Others → str()
    try:
        # Avoid treating bool as int
        if isinstance(val, float):
            return format(val, float_fmt)
        elif isinstance(val, int) and not isinstance(val, bool):
            return str(val)
        else:
            return str(val)
    except Exception:
        return str(val)


def _render_row(
        cells: Sequence[str],
        widths: Sequence[int],
        padding: int,
        _align: str,
        vbar: str = "┃",
) -> str:
    """
    Render a single visual row with vertical bars and aligned cells.
    widths are total column widths (including padding).
    """
    align_code = {"left": "<", "center": "^", "right": ">"}.get(_align, "^")

    rendered_cells: List[str] = []
    for text, width in zip(cells, widths):
        inner_w = max(0, width - 2 * padding)
        padded = f"{text:{align_code}{inner_w}}"
        cell = (" " * padding) + padded + (" " * padding)
        # Safety in case specified width is smaller than text length
        if len(cell) < width:
            cell = cell + (" " * (width - len(cell)))
        elif len(cell) > width:
            cell = cell[:width]
        rendered_cells.append(cell)

    return vbar + vbar.join(rendered_cells) + vbar
