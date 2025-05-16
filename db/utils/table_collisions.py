from sqlalchemy import select
from db.init_db import db_engine
from db_models import ProfanityClasses, SemanticClasses
from .is_empty_list import is_empty
from sqlalchemy.orm import sessionmaker


def table_collisions(table: ProfanityClasses | SemanticClasses,
                     data: ProfanityClasses | SemanticClasses) \
        -> bool | None:
    session = sessionmaker(bind=db_engine)

    with session() as session:
        if table.__tablename__ == 'profanity_classes':
            stmt = select(ProfanityClasses).where(
                ProfanityClasses.profanity_class == data.profanity_class)
            result = session.execute(stmt).scalars().first()

            return is_empty(result)
        elif table.__tablename__ == 'semantic_classes':
            stmt = select(SemanticClasses).where(
                (SemanticClasses.toxic_class == data.toxic_class) &
                (SemanticClasses.insult_class == data.insult_class) &
                (SemanticClasses.threat_class == data.threat_class) &
                (SemanticClasses.dangereous_class == data.dangereous_class)
            )
            result = session.execute(stmt).scalars().first()

            return is_empty(result)
        else:
            raise Exception('Incorrect table name')