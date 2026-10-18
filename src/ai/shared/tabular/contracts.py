"""Contracts and DTOs for scientific tabular data processing and LaTeX synthesis."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class MetricDirection(StrEnum):
    """Optimization direction defining whether higher or lower values represent superior performance."""

    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"
    NEUTRAL = "neutral"


class TabularCell(BaseModel):
    """A single metric observation in a scientific comparison matrix."""

    model_config = ConfigDict(frozen=True)

    value: float | str
    std: float | None = None
    display_text: str
    is_best: bool = False
    is_second_best: bool = False


class TabularRow(BaseModel):
    """A row representing an evaluation baseline or proposed method."""

    model_config = ConfigDict(frozen=True)

    method: str
    cells: dict[str, TabularCell] = Field(default_factory=dict)
    is_proposed_method: bool = False


class TabularMatrix(BaseModel):
    """Normalized evaluation matrix ready for scientific LaTeX rendering."""

    model_config = ConfigDict(frozen=True)

    headers: tuple[str, ...]
    rows: tuple[TabularRow, ...]
    directions: dict[str, MetricDirection] = Field(default_factory=dict)


class TableStyleOptions(BaseModel):
    """Styling preferences conforming to standard ML/CS conference guidelines."""

    model_config = ConfigDict(frozen=True)

    booktabs: bool = True
    small: bool = True
    centering: bool = True
    position: str = "t"
    precision: int = 2
    caption: str = "Experimental Results"
    label: str = "tab:results"
    notes: str | None = None
    column_alignments: str | None = None
