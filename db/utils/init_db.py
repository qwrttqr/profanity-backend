import os
import dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.exc import DatabaseError


db_engine: Engine | None = None

def connect_db():
    """
    Returns:
        db engine: Engine - database engine
    """
    dotenv.load_dotenv()
    database_con_string = os.getenv('DB_CONNECTION')
    global db_engine
    try:
        if db_engine is None:
            db_engine= create_engine(
                database_con_string,
                echo=True
            )

        return db_engine
    except:
            raise DatabaseError('Error connection to db')

def get_db_engine() -> Engine:
    """
    Returns:
        db engine: Engine - database engine
    """
    if db_engine is None:
        connect_db()

    return db_engine
