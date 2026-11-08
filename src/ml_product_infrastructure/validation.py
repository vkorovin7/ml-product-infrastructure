"""Validation helpers for project configuration examples."""


def require_non_empty(value: str, field_name: str) -> str:
    """Return a non-empty value or raise a readable validation error."""
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    return normalized
