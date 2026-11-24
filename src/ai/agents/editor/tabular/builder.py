"""LaTeX booktabs table formatter for structured scientific metrics."""

from __future__ import annotations

import re

from ai.agents.editor.tabular.contracts import TableStyleOptions, TabularMatrix

_LATEX_SPECIAL_RE = re.compile(r"(?<!\\)([_%&#])")
_MATH_SPAN_RE = re.compile(r"(\$[^$]*\$)")


def _escape_latex_text(text: str) -> str:
    """Escape unescaped LaTeX special characters (_, %, &, #) outside `$...$` math spans."""
    if not text:
        return text
    parts = _MATH_SPAN_RE.split(text)
    return "".join(
        part if idx % 2 else _LATEX_SPECIAL_RE.sub(r"\\\1", part) for idx, part in enumerate(parts)
    )


def render_latex_booktabs(
    matrix: TabularMatrix,
    options: TableStyleOptions | None = None,
) -> str:
    """Render a publication-ready LaTeX booktabs table compliant with top ML/CS conferences."""
    if not matrix.headers or not matrix.rows:
        return "% [Empty table: no headers or rows present]"

    opts = options or TableStyleOptions()
    metric_cols = matrix.headers[1:]
    col_spec = opts.column_alignments or ("l" + "c" * len(metric_cols))

    lines: list[str] = [
        f"\\begin{{table}}[{opts.position}]",
    ]

    if opts.centering:
        lines.append("  \\centering")
    if opts.small:
        lines.append("  \\small")

    clean_caption = _escape_latex_text(opts.caption.strip())
    clean_label = opts.label.strip()
    if clean_caption:
        lines.append(f"  \\caption{{{clean_caption}}}")
    if clean_label:
        lines.append(f"  \\label{{{clean_label}}}")

    lines.append(f"  \\begin{{tabular}}{{{col_spec}}}")

    # Top border
    lines.append("    \\toprule" if opts.booktabs else "    \\hline")

    # Header row
    header_cells = [f"\\textbf{{{_escape_latex_text(h)}}}" for h in matrix.headers]
    lines.append(f"    {' & '.join(header_cells)} \\\\")

    # Mid border
    lines.append("    \\midrule" if opts.booktabs else "    \\hline")

    # Data rows
    for row in matrix.rows:
        row_tokens: list[str] = []

        # Format Method cell
        safe_method = _escape_latex_text(row.method)
        if row.is_proposed_method:
            row_tokens.append(f"\\textbf{{{safe_method}}}")
        else:
            row_tokens.append(safe_method)

        # Format Metric cells
        for m in metric_cols:
            cell = row.cells.get(m)
            if cell is None:
                row_tokens.append("-")
                continue

            formatted_val = _escape_latex_text(cell.display_text)
            if cell.is_best:
                formatted_val = f"\\textbf{{{formatted_val}}}"
            elif cell.is_second_best:
                formatted_val = f"\\underline{{{formatted_val}}}"

            row_tokens.append(formatted_val)

        lines.append(f"    {' & '.join(row_tokens)} \\\\")

    # Bottom border
    lines.append("    \\bottomrule" if opts.booktabs else "    \\hline")

    lines.append("  \\end{tabular}")

    if opts.notes:
        clean_notes = opts.notes.strip()
        lines.append(f"  \\vspace{{2pt}}\\par\\footnotesize \\textit{{Note:}} {clean_notes}")

    lines.append("\\end{table}")

    return "\n".join(lines)
