from sqlmodel import create_engine, Session
from src.memory.episodic.models import MemoryItem

DATABASE_URL = "postgresql://memory_user:secure_password@localhost:5432/thinkingmemory"

engine = create_engine(DATABASE_URL)

def init_db():
    SQLModel.metadata.create_all(engine)

def get_session():
    with Session(engine) as session:
        yield session