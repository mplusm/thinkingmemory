"""
The unified agent-memory database engine.

Collapses the legacy four-layer API (working/episodic/semantic/procedural) into a
single ``Memory`` substrate whose query primitive is ``recall`` — intent in, a
ranked, token-budget-packed context window out. See ``agent-db-plan.md``.
"""
