"""Builder pattern assembling publication-quality LaTeX booktabs tables."""

from __future__ import annotations

import re

from ai.shared.tabular.contracts import TableStyleOptions, TabularMatrix

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


class LatexBooktabsBuilder:
    """Builder constructing publication-ready LaTeX tables compliant with top ML/CS conferences."""

    def __init__(
        self,
        matrix: TabularMatrix | None = None,
        options: TableStyleOptions | None = None,
    ) -> None:
        self._matrix = matrix or TabularMatrix(headers=(), rows=())
        self._options = options or TableStyleOptions()

    def build(self) -> str:
        """Render the complete LaTeX table environment string."""
        if not self._matrix.headers or not self._matrix.rows:
            return "% [Empty table: no headers or rows present]"

        metric_cols = self._matrix.headers[1:]

        # 1. Determine column alignments
        if self._options.column_alignments:
            col_spec = self._options.column_alignments
        else:
            col_spec = "l" + "c" * len(metric_cols)

        lines: list[str] = [
            f"\\begin{{table}}[{self._options.position}]",
        ]

        if self._options.centering:
            lines.append("  \\centering")
        if self._options.small:
            lines.append("  \\small")

        clean_caption = _escape_latex_text(self._options.caption.strip())
        clean_label = self._options.label.strip()
        if clean_caption:
            lines.append(f"  \\caption{{{clean_caption}}}")
        if clean_label:
            lines.append(f"  \\label{{{clean_label}}}")

        lines.append(f"  \\begin{{tabular}}{{{col_spec}}}")

        # Top border
        if self._options.booktabs:
            lines.append("    \\toprule")
        else:
            lines.append("    \\hline")

        # Header row
        header_cells = [f"\\textbf{{{_escape_latex_text(h)}}}" for h in self._matrix.headers]
        lines.append(f"    {' & '.join(header_cells)} \\\\")

        # Mid border
        if self._options.booktabs:
            lines.append("    \\midrule")
        else:
            lines.append("    \\hline")

        # Data rows
        for row in self._matrix.rows:
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
        if self._options.booktabs:
            lines.append("    \\bottomrule")
        else:
            lines.append("    \\hline")

        lines.append("  \\end{tabular}")

        if self._options.notes:
            clean_notes = self._options.notes.strip()
            lines.append(f"  \\vspace{{2pt}}\\par\\footnotesize \\textit{{Note:}} {clean_notes}")

        lines.append("\\end{table}")

        return "\n".join(lines)
