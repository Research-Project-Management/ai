"""
Server-side confirmation guard for destructive tool operations.

HOW IT WORKS
------------
1. `set_user_message(text)` is called at the chat endpoint with the raw user text.
2. Destructive tools (`delete_task`, `delete_cycle`, ...) call `confirm_gate(desc)`
   before executing.
3. If the user's message contains an explicit confirmation keyword  → proceed (returns None).
4. If not → returns a structured dict that tells the LLM "confirmation required".
5. The LLM relays this to the user as a friendly confirmation prompt.
6. On the next turn, the user types "yes" / "xác nhận" → gate passes → tool executes.

This provides server-side safety without a complex pause/resume mechanism.
No destructive operation can be executed unless the user's message contains
an explicit approval signal — even if the LLM malfunctions.
"""

from __future__ import annotations

from app.tools._client import _user_message  # shared ContextVar

# ── Confirmation vocabulary ────────────────────────────────────────────────────
# English + Vietnamese confirmation keywords (all lowercase for case-insensitive match)

_CONFIRM_KEYWORDS: frozenset[str] = frozenset({
    # English
    "yes", "yes,", "yes.", "confirm", "confirmed", "ok", "okay", "sure",
    "proceed", "go ahead", "delete it", "delete", "do it", "agree",
    # Vietnamese
    "xác nhận", "đồng ý", "chắc chắn", "tiến hành", "xóa đi", "xóa",
    "có", "ok xóa", "đồng ý xóa", "oke", "ờ", "ừ",
})

# ── Sensitive tools requiring user confirmation ────────────────────────────────
# Single source of truth — imported by orchestrator.py and any other caller.
# Add new destructive tool names here; they will be auto-protected everywhere.
CONFIRM_REQUIRED_TOOLS: frozenset[str] = frozenset({
    # Tasks
    "delete_task",
    "update_task",
    # Pages
    "delete_page",
    "update_page",
    # Cycles
    "delete_cycle",
    "update_cycle",
    # Stickies
    "delete_sticky",
    # Tags
    "delete_tag",
    # Project management
    "remove_project_member",
    "update_project",
    "update_member_role",
})


def _is_confirmed() -> bool:
    """Return True if the current user message contains an explicit confirmation."""
    text = _user_message.get().strip().lower()
    if not text:
        return False
    # Check whole-message match (short confirmations like "yes", "có")
    if text in _CONFIRM_KEYWORDS:
        return True
    # Check substring (e.g. "yes, delete the task")
    return any(kw in text for kw in _CONFIRM_KEYWORDS)


def confirm_gate(action_description: str) -> dict | None:
    """Server-side gate for destructive operations.

    Returns:
        None — if the user confirmed → caller should proceed with the operation.
        dict — if confirmation is missing → return this dict as the tool result.
                The LLM will relay it as a confirmation prompt to the user.
    """
    if _is_confirmed():
        return None  # ✅ Confirmed — proceed

    # ❌ Not confirmed — block and ask
    return {
        "needs_confirmation": True,
        "action": action_description,
        "message": (
            f"⚠️ **Confirmation required** before executing: {action_description}\n\n"
            "This action is **irreversible**. Please reply with:\n"
            "- **\"yes\"** / **\"xác nhận\"** to proceed\n"
            "- **\"no\"** / **\"cancel\"** to abort"
        ),
    }
