#!/usr/bin/env python3
"""
Run the memory lifecycle maintenance cycle.

Intended to be invoked on a schedule (cron/systemd timer) per the cadence you
pass with --interval-days. Runs decay -> consolidate -> supersede -> forget ->
prune for one agent, or for every agent if --all is given.

Usage:
    python scripts/run_lifecycle.py --agent agent-1
    python scripts/run_lifecycle.py --all --interval-days 1
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text

from thinkingmemory.core.database import get_engine
from thinkingmemory.engine import lifecycle


def _all_agents() -> list[tuple[str, str]]:
    with get_engine().connect() as conn:
        return [
            (r[0], r[1])
            for r in conn.execute(
                text("SELECT DISTINCT agent_id, tenant_id FROM memory WHERE valid_to IS NULL")
            )
        ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent")
    ap.add_argument("--tenant", default=None)
    ap.add_argument("--all", action="store_true", help="run for every agent")
    ap.add_argument("--interval-days", type=float, default=1.0)
    args = ap.parse_args()

    if args.all:
        targets = _all_agents()
    elif args.agent:
        targets = [(args.agent, args.tenant)]
    else:
        ap.error("provide --agent AGENT or --all")

    for agent_id, tenant_id in targets:
        counts = lifecycle.run_all(agent_id, tenant_id, interval_days=args.interval_days)
        print(f"{agent_id} (tenant={tenant_id}): {counts}")


if __name__ == "__main__":
    main()
