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


__all__ = ["utcnow"]
