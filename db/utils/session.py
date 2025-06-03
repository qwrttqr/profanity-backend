from requests import Session
from sqlalchemy.orm import sessionmaker
from .init_db import get_db_engine

def get_session() -> Session:
    """

    Returns:
        Session: Session - fabric for creating new sessions
    """
    engine = get_db_engine()
    Session = sessionmaker(bind=engine)

    return Session