"""Tests for per-tenant Row-Level Security (database-enforced isolation)."""

from sqlalchemy import text

from thinkingmemory.engine import store
from thinkingmemory.core.database import enable_rls, disable_rls, get_engine


def test_rls_enforces_isolation_at_db_level(db_available, agent_id, cleanup_agent):
    """A raw query under tenant A must not see tenant B's rows — no app WHERE."""
    cleanup_agent(agent_id)
    store.remember(agent_id, {"text": "alpha — tenant A data"}, tenant_id="rlsA")
    store.remember(agent_id, {"text": "bravo — tenant B data"}, tenant_id="rlsB")

    enable_rls()
    try:
        eng = get_engine()
        # GUC unset -> policy allows all (single-tenant / admin behavior)
        with eng.begin() as conn:
            rows = conn.execute(
                text("SELECT tenant_id FROM memory WHERE agent_id = :a"), {"a": agent_id}
            ).fetchall()
            assert {r[0] for r in rows} == {"rlsA", "rlsB"}

        # GUC = rlsA -> Postgres hides tenant B even though the query has no
        # tenant filter. This is enforcement below the application layer.
        with eng.begin() as conn:
            conn.execute(text("SELECT set_config('app.tenant_id', 'rlsA', true)"))
            rows = conn.execute(
                text("SELECT tenant_id FROM memory WHERE agent_id = :a"), {"a": agent_id}
            ).fetchall()
            assert {r[0] for r in rows} == {"rlsA"}
    finally:
        disable_rls()


def test_rls_insert_check(db_available, agent_id, cleanup_agent):
    """Under a tenant GUC, you cannot insert a row for a different tenant."""
    cleanup_agent(agent_id)
    enable_rls()
    try:
        eng = get_engine()
        with eng.begin() as conn:
            conn.execute(text("SELECT set_config('app.tenant_id', 'rlsA', true)"))
            # inserting a row tagged rlsB while scoped to rlsA must fail the policy
            failed = False
            try:
                conn.execute(
                    text(
                        "INSERT INTO memory (tenant_id, agent_id, scope, mtype, content, text, "
                        "salience, confidence, decay_rate, recall_count, valid_from, created_at) "
                        "VALUES ('rlsB', :a, 'private', 'episodic', '{}', 'x', "
                        "1.0, 1.0, 0.0, 0, now(), now())"
                    ),
                    {"a": agent_id},
                )
            except Exception:
                failed = True
            assert failed, "RLS WITH CHECK should reject cross-tenant insert"
    finally:
        disable_rls()
