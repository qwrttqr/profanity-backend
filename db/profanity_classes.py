from db.init_db import db_engine
from db.utils import table_collisions
from db_models import ProfanityClasses
from sqlalchemy.orm import sessionmaker


def save_to_profanity_classes(profanity_class: int):
    new_profanity_class = ProfanityClasses(profanity_class=profanity_class)
    session = sessionmaker(db_engine)
    if table_collisions(table=ProfanityClasses,
                        data=new_profanity_class):
        raise Exception('Profanity class already exists')
    else:
        with session() as ss:
            ss.add(new_profanity_class)
            ss.commit()
        return new_profanity_class.id
