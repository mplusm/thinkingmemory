#!/usr/bin/env python3
"""Drop and recreate all database tables."""

import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text

from thinkingmemory.core.database import get_engine, init_db

if __name__ == "__main__":
    engine = get_engine()

    # Drop all tables
    with engine.connect() as connection:
        connection.execute(text("DROP TABLE IF EXISTS memoryitem CASCADE"))
        connection.execute(text("DROP TABLE IF EXISTS fact CASCADE"))
        connection.execute(text("DROP TABLE IF EXISTS procedure CASCADE"))
        connection.execute(text("DROP TABLE IF EXISTS datasource CASCADE"))
        connection.execute(text("DROP TABLE IF EXISTS datatable CASCADE"))
        connection.execute(text("DROP TABLE IF EXISTS datacolumn CASCADE"))
        connection.execute(text("DROP TABLE IF EXISTS knowledgeentity CASCADE"))
        connection.execute(text("DROP TABLE IF EXISTS userpreference CASCADE"))
        connection.execute(text("DROP TABLE IF EXISTS workflowhabit CASCADE"))
        connection.commit()

    print("Dropped all tables successfully!")

    # Recreate all tables
    init_db()

    print("Recreated all tables successfully!")
