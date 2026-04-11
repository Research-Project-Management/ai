"""User tools — search users and get profile information."""

from __future__ import annotations

from langchain_core.tools import tool

from app.tools._client import flux_api


@tool
async def get_current_user() -> dict:
    """Get the profile of the currently logged-in user.

    Use this when the user asks:
    - "thông tin tài khoản của tôi"
    - "profile của tôi"
    - "tôi đang dùng tài khoản nào?"

    Returns name, email, avatar, and user ID.
    """
    return await flux_api.get("/auth/user")


@tool
async def search_users(query: str) -> dict:
    """Search for users by name or email across the platform.

    Use this when you need to find a user's ID to assign tasks, add to projects,
    or look up team member information.

    Args:
        query: Search term — name or email (partial match supported).
    """
    return await flux_api.get("/auth/search", params={"q": query})
