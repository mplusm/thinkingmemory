#!/usr/bin/env python3
"""Drop and recreate all database tables."""

import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlmodel import create_engine
from sqlalchemy import text
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://memory_user:postgres#2026@localhost:5432/thinkingmemory")

engine = create_engine(DATABASE_URL)

# Drop all tables
with engine.connect() as connection:
    connection.execute(text("DROP TABLE IF EXISTS memoryitem CASCADE"))
    connection.execute(text("DROP TABLE IF EXISTS fact CASCADE"))
    connection.execute(text("DROP TABLE IF EXISTS procedure CASCADE"))
    connection.commit()

print("Dropped all tables successfully!")

# Recreate tables
from src.memory.episodic.database import init_db as init_episodic_db
from src.memory.semantic.database import init_db as init_semantic_db
from src.memory.procedural.database import init_db as init_procedural_db

init_episodic_db()
init_semantic_db()
init_procedural_db()

print("Recreated all tables successfully!")