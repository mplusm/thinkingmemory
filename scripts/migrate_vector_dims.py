#!/usr/bin/env python3
"""
Migrate existing unsized `vector` embedding columns to `vector(EMBEDDING_DIM)`
and create the HNSW similarity indexes.

Use this on a database created before embedding dimensions were pinned. It is
non-destructive: only columns without a declared dimension are altered, and the
HNSW indexes are created with IF NOT EXISTS.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from thinkingmemory.config.settings import get_settings
from thinkingmemory.core.database import migrate_vector_columns

if __name__ == "__main__":
    dim = get_settings().embedding_dim
    print(f"Migrating embedding columns to vector({dim}) and building HNSW indexes...")
    migrate_vector_columns(dim)
    print("Vector columns migrated and indexes created.")
