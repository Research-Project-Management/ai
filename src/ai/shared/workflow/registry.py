"""Generic tool registry shared by tool-using agents."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Protocol

from pydantic import BaseModel


class ToolSpec(Protocol):
    """Structural view of a tool spec shared across registry, gate, and executor."""

    @property
    def name(self) -> str: ...
    @property
    def description(self) -> str: ...
    @property
    def risk_tier(self) -> str: ...
    @property
    def idempotent(self) -> bool: ...
    @property
    def args_schema(self) -> type[BaseModel]: ...


def format_args_hint(schema_cls: type[BaseModel]) -> str:
    """Render `field*: type` hints for planner catalogs (`*` marks required fields)."""
    parts: list[str] = []
    for field_name, field_info in schema_cls.model_fields.items():
        req = "*" if field_info.is_required() else ""
        ann = getattr(field_info.annotation, "__name__", str(field_info.annotation))
        parts.append(f"{field_name}{req}: {ann.replace('typing.', '')}")
    return ", ".join(parts) if parts else "(no args)"


class ToolRegistry[S: ToolSpec]:
    """Ordered, duplicate-checked catalog of tool specs used for planning and policy gates."""

    label: str = "tool"

    def __init__(self, specs: Iterable[S] = ()) -> None:
        ordered: list[S] = []
        by_name: dict[str, S] = {}
        for spec in specs:
            if spec.name in by_name:
                raise ValueError(f"Duplicate {self.label} tool: {spec.name}")
            ordered.append(spec)
            by_name[spec.name] = spec
        self._specs: tuple[S, ...] = tuple(ordered)
        self._by_name: dict[str, S] = by_name

    @property
    def specs(self) -> tuple[S, ...]:
        return self._specs

    @property
    def names(self) -> frozenset[str]:
        return frozenset(self._by_name)

    def get(self, name: str) -> S | None:
        return self._by_name.get(name)

    def format_catalog(self) -> str:
        return "\n".join(
            f"- `{s.name}` [{s.risk_tier}] ({format_args_hint(s.args_schema)}): {s.description}"
            for s in self._specs
        )

    def to_openai_tools(self) -> list[dict[str, Any]]:
        """Convert all registered specs into OpenAI function-calling tool schemas."""
        return [
            {
                "type": "function",
                "function": {
                    "name": s.name,
                    "description": s.description,
                    "parameters": s.args_schema.model_json_schema(),
                },
            }
            for s in self._specs
        ]
