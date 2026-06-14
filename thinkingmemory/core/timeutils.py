"""
Time helpers.

``datetime.utcnow()`` is deprecated in Python 3.12+. The database ``timestamp``
columns are timezone-naive, so this returns a naive UTC datetime to preserve the
existing storage semantics while avoiding the deprecation warning.
"""

from datetime import datetime, timezone


def utcnow() -> datetime:
    """Naive UTC timestamp (no tzinfo), matching the DB columns."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def to_naive_utc(dt: datetime) -> datetime:
    """Normalize a datetime to naive UTC so it compares with the DB columns.

    Timezone-aware inputs (e.g. an API caller's ISO timestamp with offset) are
    converted to UTC and stripped of tzinfo; naive inputs pass through.
    """
    if dt.tzinfo is not None:
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


__all__ = ["utcnow", "to_naive_utc"]
