from db.init_db import db_engine
from db_models import Answer
from sqlalchemy.orm import sessionmaker

def save_to_answer(semantic_info: list):

    new_answer: Answer = Answer()
    session = sessionmaker(db_engine)

    with session() as ss:
        ss.add(new_answer)
        ss.commit()
    return new_answer.id