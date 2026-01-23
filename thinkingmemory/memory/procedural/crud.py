from sqlmodel import select, delete
from sqlalchemy import func

from thinkingmemory.core.database import get_session_context
from thinkingmemory.memory.procedural.models import Procedure

def store_procedure(agent_id: str, name: str, description: str = None, steps: list[dict] = None, success_rate: float = 1.0, version: int = 1):
    procedure = Procedure(
        agent_id=agent_id,
        name=name,
        description=description,
        steps=steps,
        success_rate=success_rate,
        version=version
    )
    with get_session_context() as session:
        session.add(procedure)
        session.commit()
        session.refresh(procedure)
        return procedure

def retrieve_procedures(agent_id: str, limit: int = 10):
    with get_session_context() as session:
        statement = select(Procedure).where(Procedure.agent_id == agent_id).limit(limit)
        return session.exec(statement).all()

def update_procedure_success_rate(procedure_id: int, success_rate: float):
    with get_session_context() as session:
        procedure = session.get(Procedure, procedure_id)
        if procedure:
            procedure.success_rate = success_rate
            session.commit()
            session.refresh(procedure)
        return procedure

def forget_low_success_procedures(agent_id: str, success_threshold: float = 0.5):
    """Delete procedures with success rate below the threshold."""
    with get_session_context() as session:
        statement = delete(Procedure).where(
            Procedure.agent_id == agent_id,
            Procedure.success_rate < success_threshold
        )
        result = session.exec(statement)
        session.commit()
        return result.rowcount

def retrieve_similar_procedures(agent_id: str, embedding: list[float], limit: int = 10, similarity_threshold: float = 0.5):
    """Retrieve procedures similar to the given embedding."""
    with get_session_context() as session:
        # Calculate cosine distance between the query embedding and stored embeddings
        # pgvector uses <-> for cosine distance (lower is more similar)
        # We'll use the <-> operator via func
        statement = select(Procedure).where(
            Procedure.agent_id == agent_id,
            Procedure.embedding.isnot(None)
        ).order_by(
            func.l2_distance(Procedure.embedding, embedding)
        ).limit(limit)
        return session.exec(statement).all()