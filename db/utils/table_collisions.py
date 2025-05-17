from sqlalchemy import select
from db.db_models import ProfanityClasses, SemanticClasses
from sqlalchemy.orm import sessionmaker
from db.utils.session import get_session


def table_collisions(table: ProfanityClasses | SemanticClasses,
                     data: ProfanityClasses | SemanticClasses) \
        -> int | None:
    LocalSession = get_session()

    with LocalSession() as ss:
        if table.__tablename__ == 'profanity_classes':
            stmt = select(ProfanityClasses).where(
                ProfanityClasses.profanity_class == data.profanity_class)
            result = ss.execute(stmt).scalars().first()
            ss.close()
            if result:
                return result.id
            else:
                return None

        elif table.__tablename__ == 'semantic_classes':
            stmt = select(SemanticClasses).where(
                (SemanticClasses.toxic_class == data.toxic_class) &
                (SemanticClasses.insult_class == data.insult_class) &
                (SemanticClasses.threat_class == data.threat_class) &
                (SemanticClasses.dangerous_class == data.dangerous_class)
            )
            result = ss.execute(stmt).scalars().first()
            ss.close()
            if result:

                return result.id
            else:
                return None
        else:
            raise Exception('Incorrect table name')