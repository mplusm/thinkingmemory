#!/usr/bin/env python3
"""
Enable (or disable) per-tenant Row-Level Security on the memory tables.

With RLS on, Postgres itself restricts every query to the tenant set in the
session's ``app.tenant_id`` GUC — defense-in-depth beneath the app's tenant
filtering. The policy allows access when the GUC is unset (single-tenant /
admin / maintenance), so enabling it is safe for existing operations.

Set ``RLS_ENABLED=true`` in the environment so the app sets the GUC per request.

Usage:
    python scripts/enable_rls.py            # enable
    python scripts/enable_rls.py --disable  # disable
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from thinkingmemory.core.database import enable_rls, disable_rls, init_db

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--disable", action="store_true")
    args = ap.parse_args()

    init_db()  # ensure tables exist
    if args.disable:
        disable_rls()
        print("Row-Level Security disabled.")
    else:
        enable_rls()
        print("Row-Level Security enabled. Set RLS_ENABLED=true so the app sets app.tenant_id.")
