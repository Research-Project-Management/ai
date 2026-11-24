"""Sub-client handling project analytics endpoints."""

from __future__ import annotations

from ai.agents.project.gateway.contracts import (
    ProjectAnalytics,
    ProjectGatewayError,
    ProjectScope,
)
from ai.shared.gateway.transport import BackendTransport, encode_seg


class AnalyticsClient:
    """Adapter for `/api/v1/projects/:id/analytics`."""

    def __init__(self, transport: BackendTransport) -> None:
        self._transport = transport

    async def get_project_analytics(self, scope: ProjectScope, project_id: str) -> ProjectAnalytics:
        del scope
        raw = await self._transport.request(
            "GET",
            f"/api/v1/projects/{encode_seg(project_id)}/analytics",
        )
        if not isinstance(raw, dict):
            raise ProjectGatewayError(502, "Unexpected analytics payload format from Backend")

        state_groups_raw = raw.get("stateGroups")
        priority_raw = raw.get("priority")
        return ProjectAnalytics(
            project_id=project_id,
            total_items=int(raw.get("totalItems") or 0),
            completed_items=int(raw.get("completedItems") or 0),
            in_progress_items=int(raw.get("inProgressItems") or 0),
            overdue_items=int(raw.get("overdueItems") or 0),
            unassigned_items=int(raw.get("unassignedItems") or 0),
            completion_rate=float(raw.get("completionRate") or 0.0),
            state_groups=(
                {str(k): int(v) for k, v in state_groups_raw.items() if isinstance(v, int | float)}
                if isinstance(state_groups_raw, dict)
                else {}
            ),
            states=(
                [s for s in raw["states"] if isinstance(s, dict)]
                if isinstance(raw.get("states"), list)
                else []
            ),
            priority=(
                {str(k): int(v) for k, v in priority_raw.items() if isinstance(v, int | float)}
                if isinstance(priority_raw, dict)
                else {}
            ),
            assignee=(
                [a for a in raw["assignee"] if isinstance(a, dict)]
                if isinstance(raw.get("assignee"), list)
                else []
            ),
            labels=(
                [lb for lb in raw["labels"] if isinstance(lb, dict)]
                if isinstance(raw.get("labels"), list)
                else []
            ),
        )
