import os
import dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.exc import DatabaseError


db_engine: Engine | None = None

def connect_db():
    '''
    creates db engine.
    :return: Engine
    '''
    dotenv.load_dotenv()
    database_username = os.environ.get('DB_USERNAME')
    database_password = os.environ.get('DB_PASSWORD')
    global db_engine
    try:
        if db_engine is None:
            db_engine= create_engine(
                f'mysql+pymysql://{database_username}:'
                f'{database_password}@localhost:3306/profanity-neuro-db',
                echo=True
            )
        return db_engine
    except:
            raise DatabaseError('Error connection to db')

def get_db_engine() -> Engine:
    '''
    gets db engine.
    :return: Engine
    :return:
    '''
    if db_engine is None:
        connect_db()
    return db_engine
