import os
import dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


def connect_db():
    dotenv.load_dotenv()
    database_username = os.environ.get('DB_USERNAME')
    database_password = os.environ.get('DB_PASSWORD')
    try:
        engine = create_engine(
            f'mysql+pymysql://{database_username}:{database_password}@localhost:3306/profantiy-neuro-db'
        )
        return engine
    except:
        print('Error connection to db')


database_engine = connect_db()

Session = sessionmaker(bind=database_engine)
