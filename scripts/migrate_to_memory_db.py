#!/usr/bin/env python3
"""
Migrate the legacy four-layer tables into the unified ``memory`` table.

Reads legacy rows via raw SQL (legacy embeddings are all NULL, so we re-embed
each row's text with the configured provider) and inserts them as ``Memory``
rows with the appropriate ``mtype`` policy tag. Legacy tables are left in place.

Idempotent-ish: pass --fresh to clear previously migrated rows (those whose
provenance.source == 'legacy_migration') before re-importing.

Usage:
    python scripts/migrate_to_memory_db.py [--fresh] [--batch 256]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text

from thinkingmemory.core.database import get_engine, init_db
from thinkingmemory.engine import store


def _rows(conn, sql):
    return [dict(r._mapping) for r in conn.execute(text(sql))]


def _render_legacy_text(mtype: str, row: dict) -> str:
    """Build an embeddable text string from a legacy row."""
    if mtype == "episodic":  # memoryitem.content (JSON)
        c = row.get("content")
        if isinstance(c, dict):
            return c.get("text") or "\n".join(f"{k}: {v}" for k, v in c.items())
        return str(c)
    if "fact" in row:  # fact
        return row["fact"]
    if "name" in row and "steps" in row:  # procedure
        return f"{row.get('name','')}: {row.get('description') or ''}".strip()
    if "category" in row:  # userpreference
        return f"{row['category']}/{row['key']}: {row['value']}"
    if "habit_name" in row:  # workflowhabit
        return f"{row['habit_name']}: {row.get('description') or row.get('pattern')}"
    if "entity_type" in row:  # knowledgeentity
        return f"{row['entity_type']} {row['name']}: {row.get('description') or ''}".strip()
    # data* tables
    name = row.get("source_name") or row.get("table_name") or row.get("column_name") or ""
    return f"{name}: {row.get('description') or ''}".strip()


# (legacy table, mtype, content-builder)
LEGACY = [
    ("memoryitem", "episodic", lambda r: r.get("content") or {}),
    ("fact", "semantic", lambda r: {"fact": r["fact"], "confidence": r.get("confidence")}),
    ("procedure", "procedural", lambda r: {"name": r.get("name"), "steps": r.get("steps"), "description": r.get("description")}),
    ("userpreference", "procedural", lambda r: {"category": r.get("category"), "key": r.get("key"), "value": r.get("value")}),
    ("workflowhabit", "procedural", lambda r: {"habit_name": r.get("habit_name"), "pattern": r.get("pattern")}),
    ("knowledgeentity", "semantic", lambda r: {"entity_type": r.get("entity_type"), "name": r.get("name"), "properties": r.get("properties")}),
    ("datasource", "semantic", lambda r: {"source_name": r.get("source_name"), "description": r.get("description")}),
    ("datatable", "semantic", lambda r: {"table_name": r.get("table_name"), "description": r.get("description")}),
    ("datacolumn", "semantic", lambda r: {"column_name": r.get("column_name"), "description": r.get("description")}),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", action="store_true", help="clear prior migrated rows first")
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    init_db()  # ensure the memory table + indexes exist
    engine = get_engine()

    if args.fresh:
        with engine.begin() as conn:
            n = conn.execute(
                text("DELETE FROM memory WHERE provenance->>'source' = 'legacy_migration'")
            ).rowcount
        print(f"Cleared {n} previously migrated rows.")

    total = 0
    for table, mtype, content_fn in LEGACY:
        with engine.connect() as conn:
            try:
                rows = _rows(conn, f"SELECT * FROM {table}")
            except Exception as exc:
                print(f"  skip {table}: {exc}")
                continue

        batch = []
        for r in rows:
            content = content_fn(r)
            batch.append(
                {
                    "agent_id": r.get("agent_id", "default"),
                    "content": content if isinstance(content, dict) else {"value": content},
                    "text": _render_legacy_text(mtype, r),
                    "mtype": mtype,
                    "confidence": r.get("confidence", 1.0) or 1.0,
                    "tenant_id": r.get("tenant_id", "default"),
                    "provenance": {"source": "legacy_migration", "legacy_table": table, "legacy_id": r.get("id")},
                }
            )
            if len(batch) >= args.batch:
                store.remember_many(batch)
                total += len(batch)
                batch = []
        if batch:
            store.remember_many(batch)
            total += len(batch)
        print(f"  {table:16} -> {mtype:10} ({len(rows)} rows)")

    print(f"Migrated {total} legacy rows into the memory table.")


if __name__ == "__main__":
    main()
