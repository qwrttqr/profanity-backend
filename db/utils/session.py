from requests import Session
from sqlalchemy.orm import sessionmaker
from .init_db import get_db_engine

def get_session() -> Session:
    '''
    returns session fabric class
    :return: Session
    '''
    engine = get_db_engine()
    Session = sessionmaker(bind=engine)
    return Session