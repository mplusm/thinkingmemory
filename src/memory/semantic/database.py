from sqlmodel import create_engine, Session, SQLModel
from src.memory.semantic.models import Fact
from dotenv import load_dotenv
import os

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://memory_user:postgres#2026@localhost:5432/thinkingmemory")

engine = create_engine(DATABASE_URL)

def init_db():
    SQLModel.metadata.create_all(engine)

def get_session():
    with Session(engine) as session:
        yield session