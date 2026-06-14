#!/usr/bin/env python3
"""
Convert an existing non-partitioned `memory` table to HASH partitioning by
tenant_id, preserving all rows and ids.

Non-destructive within a transaction: rename the old table aside, create the
partitioned table, copy rows (preserving identity ids), fix the sequence, drop
the old table, recreate indexes, and re-apply RLS if it was enabled. Run in a
maintenance window (stop the service first).

Usage:
    python scripts/partition_memory.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text

from thinkingmemory.config.settings import get_settings
from thinkingmemory.core.database import (
    get_engine,
    create_memory_partitioned,
    create_memory_indexes,
    enable_rls,
)

_COLUMNS = (
    "id, tenant_id, agent_id, scope, mtype, content, text, embedding, salience, "
    "confidence, decay_rate, recall_count, last_recalled_at, valid_from, valid_to, "
    "created_at, superseded_at, provenance"
)


def main():
    engine = get_engine()
    with engine.connect() as conn:
        already = conn.execute(
            text("SELECT relkind FROM pg_class WHERE relname = 'memory'")
        ).scalar()
        if already == "p":
            print("`memory` is already partitioned — nothing to do.")
            return
        rls_on = bool(
            conn.execute(
                text("SELECT relrowsecurity FROM pg_class WHERE relname = 'memory'")
            ).scalar()
        )
        n_before = conn.execute(text("SELECT count(*) FROM memory")).scalar()

    print(f"Migrating {n_before} rows to a partitioned table (RLS was {'on' if rls_on else 'off'})...")
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE memory RENAME TO memory_old"))

    # Create the fresh partitioned table (now that `memory` is out of the way).
    create_memory_partitioned()

    with engine.begin() as conn:
        conn.execute(
            text(
                f"INSERT INTO memory ({_COLUMNS}) OVERRIDING SYSTEM VALUE "
                f"SELECT {_COLUMNS} FROM memory_old"
            )
        )
        # Advance the identity sequence past the largest migrated id.
        conn.execute(
            text(
                "SELECT setval(pg_get_serial_sequence('memory', 'id'), "
                "GREATEST((SELECT max(id) FROM memory), 1))"
            )
        )
        conn.execute(text("DROP TABLE memory_old CASCADE"))

    create_memory_indexes()
    if rls_on:
        enable_rls()

    with engine.connect() as conn:
        n_after = conn.execute(text("SELECT count(*) FROM memory")).scalar()
        parts = conn.execute(
            text("SELECT count(*) FROM pg_inherits WHERE inhparent = 'memory'::regclass")
        ).scalar()
    print(f"Done: {n_after} rows across {parts} partitions (expected {n_before}).")
    assert n_after == n_before, "row count mismatch after migration!"


if __name__ == "__main__":
    main()
