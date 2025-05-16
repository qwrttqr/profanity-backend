from db.init_db import db_engine
from sqlalchemy.orm import sessionmaker
from db.utils import table_collisions
from db_models import SemanticClasses

def save_to_semantic_classes(semantic_info: list):
    toxic_class = semantic_info[0]
    insult_class = semantic_info[1]
    threat_class= semantic_info[2]
    dangerous_class = semantic_info[3]
    new_semantic_class = SemanticClasses(toxic_class=toxic_class,
                                          insult_class=insult_class,
                                          threat_class=threat_class,
                                          dangerous_class=dangerous_class)
    if table_collisions(table=SemanticClasses,
                        data=new_semantic_class):
        raise Exception('Semantic class already exists')
    else:
        session = sessionmaker(db_engine)
        with session() as ss:
            ss.add(new_semantic_class)
            ss.commit()
        return new_semantic_class.id