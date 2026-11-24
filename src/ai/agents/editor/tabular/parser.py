"""Data parser transforming raw CSV, TSV, or JSON metric outputs into a TabularMatrix."""

from __future__ import annotations

import csv
import io
import json
import math
from collections import defaultdict
from typing import Any

from ai.agents.editor.tabular.contracts import (
    MetricDirection,
    TabularCell,
    TabularMatrix,
    TabularRow,
)
from ai.agents.editor.tabular.heuristics import infer_metric_direction, is_proposed_method_name


def _parse_float(raw: Any) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, int | float):
        return float(raw)
    cleaned = str(raw).strip().replace("%", "").replace(",", "")
    try:
        val = float(cleaned)
        return None if math.isnan(val) or math.isinf(val) else val
    except ValueError:
        return None


def _calculate_stats(values: list[float]) -> tuple[float, float | None]:
    if not values:
        return 0.0, None
    mean = sum(values) / len(values)
    if len(values) > 1:
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        std = math.sqrt(variance)
        return mean, std
    return mean, None


class TabularDataParser:
    """Parses raw scientific tabular data (CSV/JSON/TSV) and calculates aggregated rankings."""

    def __init__(self, precision: int = 2) -> None:
        self.precision = precision

    def parse_csv(
        self,
        raw_text: str,
        *,
        delimiter: str = ",",
        direction_overrides: dict[str, MetricDirection] | None = None,
    ) -> TabularMatrix:
        """Parse delimited text (CSV/TSV) into a ranked TabularMatrix."""
        reader = csv.reader(io.StringIO(raw_text.strip()), delimiter=delimiter)
        raw_rows = [row for row in reader if row and any(cell.strip() for cell in row)]
        if not raw_rows:
            return TabularMatrix(headers=(), rows=())

        headers = [h.strip() for h in raw_rows[0]]
        dict_rows: list[dict[str, str]] = []
        for r in raw_rows[1:]:
            row_dict = {headers[i]: r[i].strip() if i < len(r) else "" for i in range(len(headers))}
            dict_rows.append(row_dict)

        return self.parse_records(
            dict_rows,
            headers=headers,
            direction_overrides=direction_overrides,
        )

    def parse_json(
        self,
        raw_json: str | list[dict[str, Any]] | dict[str, Any],
        *,
        direction_overrides: dict[str, MetricDirection] | None = None,
    ) -> TabularMatrix:
        """Parse JSON metric objects or lists into a TabularMatrix."""
        if isinstance(raw_json, str):
            payload = json.loads(raw_json)
        else:
            payload = raw_json

        records: list[dict[str, Any]] = []
        if isinstance(payload, list):
            records = [r for r in payload if isinstance(r, dict)]
        elif isinstance(payload, dict):
            # Check if dict of {method: {metric: val}} or {results: [...]}
            if "results" in payload and isinstance(payload["results"], list):
                records = [r for r in payload["results"] if isinstance(r, dict)]
            elif "experiments" in payload and isinstance(payload["experiments"], list):
                records = [r for r in payload["experiments"] if isinstance(r, dict)]
            else:
                for k, v in payload.items():
                    if isinstance(v, dict):
                        records.append({"method": k, **v})
                    else:
                        records.append({"metric": k, "value": v})

        if not records:
            return TabularMatrix(headers=(), rows=())

        # Collect unique headers preserving insertion order
        headers_set: dict[str, None] = {}
        for r in records:
            for k in r:
                headers_set[k] = None
        headers = list(headers_set.keys())

        return self.parse_records(
            records,
            headers=headers,
            direction_overrides=direction_overrides,
        )

    def parse_records(
        self,
        records: list[dict[str, Any]],
        *,
        headers: list[str],
        direction_overrides: dict[str, MetricDirection] | None = None,
    ) -> TabularMatrix:
        """Core parsing and ranking logic across dictionaries of rows."""
        if not records or not headers:
            return TabularMatrix(headers=(), rows=())

        # Determine method / row label column
        method_candidates = (
            "method",
            "model",
            "system",
            "name",
            "architecture",
            "baseline",
            "approach",
        )
        excluded_meta_cols = frozenset(
            {"seed", "run", "run_id", "trial", "fold", "split", "step", "epoch", "id", "index"}
        )
        method_col = next((h for h in headers if h.lower() in method_candidates), headers[0])
        metric_cols = [
            h for h in headers if h != method_col and h.strip().lower() not in excluded_meta_cols
        ]

        # Group by method name to support multi-seed aggregations
        grouped_metrics: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        raw_strings: dict[str, dict[str, str]] = defaultdict(dict)
        method_order: list[str] = []

        for r in records:
            raw_method = str(r.get(method_col, "")).strip() or "Unnamed"
            if raw_method not in grouped_metrics:
                method_order.append(raw_method)
            for m in metric_cols:
                raw_val = r.get(m)
                parsed_num = _parse_float(raw_val)
                if parsed_num is not None:
                    grouped_metrics[raw_method][m].append(parsed_num)
                else:
                    raw_strings[raw_method][m] = str(raw_val or "").strip()

        # Determine directions
        directions: dict[str, MetricDirection] = {}
        overrides = direction_overrides or {}
        for m in metric_cols:
            directions[m] = overrides.get(m, infer_metric_direction(m))

        # Build intermediate numeric cells to compute best / 2nd best
        metric_means: dict[str, list[tuple[str, float]]] = defaultdict(list)
        stats_by_method: dict[str, dict[str, tuple[float, float | None]]] = defaultdict(dict)

        for method in method_order:
            for m in metric_cols:
                nums = grouped_metrics[method].get(m, [])
                if nums:
                    mean_val, std_val = _calculate_stats(nums)
                    stats_by_method[method][m] = (mean_val, std_val)
                    metric_means[m].append((method, mean_val))

        # Rank each metric column
        best_by_metric: dict[str, set[str]] = defaultdict(set)
        second_best_by_metric: dict[str, set[str]] = defaultdict(set)

        for m, dir_mode in directions.items():
            if dir_mode == MetricDirection.NEUTRAL:
                continue
            candidates = metric_means.get(m, [])
            if not candidates:
                continue

            reverse = dir_mode == MetricDirection.MAXIMIZE
            sorted_candidates = sorted(candidates, key=lambda x: x[1], reverse=reverse)
            top_val = sorted_candidates[0][1]

            # Collect all methods that achieve the top value
            for meth, val in sorted_candidates:
                if math.isclose(val, top_val, abs_tol=1e-6):
                    best_by_metric[m].add(meth)

            # Find second best
            second_candidates = [
                (meth, val) for meth, val in sorted_candidates if meth not in best_by_metric[m]
            ]
            if second_candidates:
                second_val = second_candidates[0][1]
                for meth, val in second_candidates:
                    if math.isclose(val, second_val, abs_tol=1e-6):
                        second_best_by_metric[m].add(meth)

        # Assemble final TabularRows
        rows: list[TabularRow] = []
        for method in method_order:
            cells: dict[str, TabularCell] = {}
            for m in metric_cols:
                if m in stats_by_method[method]:
                    mean_val, std_val = stats_by_method[method][m]
                    if std_val is not None and std_val > 0.0:
                        disp = f"{mean_val:.{self.precision}f} $\\pm$ {std_val:.{self.precision}f}"
                    else:
                        disp = f"{mean_val:.{self.precision}f}"
                    is_best = method in best_by_metric[m]
                    is_second = method in second_best_by_metric[m]
                    cells[m] = TabularCell(
                        value=mean_val,
                        std=std_val,
                        display_text=disp,
                        is_best=is_best,
                        is_second_best=is_second,
                    )
                else:
                    raw_str = raw_strings[method].get(m, "-")
                    cells[m] = TabularCell(
                        value=raw_str,
                        std=None,
                        display_text=raw_str,
                        is_best=False,
                        is_second_best=False,
                    )

            rows.append(
                TabularRow(
                    method=method,
                    cells=cells,
                    is_proposed_method=is_proposed_method_name(method),
                )
            )

        final_headers = (method_col, *metric_cols)
        return TabularMatrix(
            headers=tuple(final_headers),
            rows=tuple(rows),
            directions=directions,
        )
