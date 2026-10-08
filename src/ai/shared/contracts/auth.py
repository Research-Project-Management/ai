"""Request-scoped delegation token context shared across downstream gateway & file fetchers."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_DELEGATION_TOKEN: ContextVar[str] = ContextVar("flux_delegation_token", default="")


def get_delegation_token() -> str:
    """Return the active request's delegation bearer token (empty string if unset)."""
    return _DELEGATION_TOKEN.get()


@contextmanager
def bind_delegation_token(token: str) -> Iterator[None]:
    """Bind a user delegation bearer token for the active async context without persisting it."""
    clean = token.strip()
    if clean[:7].lower() == "bearer ":
        clean = clean[7:].strip()
    reset_token = _DELEGATION_TOKEN.set(clean)
    try:
        yield
    finally:
        _DELEGATION_TOKEN.reset(reset_token)
