from sqlmodel import select, delete
from src.memory.procedural.models import Procedure
from src.memory.procedural.database import get_session

def store_procedure(agent_id: str, name: str, description: str = None, steps: list[dict] = None, success_rate: float = 1.0, version: int = 1):
    procedure = Procedure(
        agent_id=agent_id,
        name=name,
        description=description,
        steps=steps,
        success_rate=success_rate,
        version=version
    )
    with next(get_session()) as session:
        session.add(procedure)
        session.commit()
        session.refresh(procedure)
        return procedure

def retrieve_procedures(agent_id: str, limit: int = 10):
    with next(get_session()) as session:
        statement = select(Procedure).where(Procedure.agent_id == agent_id).limit(limit)
        return session.exec(statement).all()

def update_procedure_success_rate(procedure_id: int, success_rate: float):
    with next(get_session()) as session:
        procedure = session.get(Procedure, procedure_id)
        if procedure:
            procedure.success_rate = success_rate
            session.commit()
            session.refresh(procedure)
        return procedure

def forget_low_success_procedures(agent_id: str, success_threshold: float = 0.5):
    """Delete procedures with success rate below the threshold."""
    with next(get_session()) as session:
        statement = delete(Procedure).where(
            Procedure.agent_id == agent_id,
            Procedure.success_rate < success_threshold
        )
        result = session.exec(statement)
        session.commit()
        return result.rowcount