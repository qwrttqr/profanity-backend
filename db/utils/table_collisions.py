from sqlalchemy import select
from db.db_models import ProfanityClasses, SemanticClasses, Answer
from sqlalchemy.orm import sessionmaker
from db.utils.session import get_session


def table_collisions(table: ProfanityClasses | SemanticClasses | Answer,
                     data: ProfanityClasses | SemanticClasses | Answer) -> int | None:
    """
    Finds rows in table that will correspond current set of classes.
    Returns None is there is no row with that set or int with id which row
    with that set contain.
    Args:
        table: SQLAlchemy table type
        data: Object constructed from SQLAlchemy table type
    Returns:
        id: (int | None) - if id exist returns it otherwise returns None
    """
    LocalSession = get_session()

    with LocalSession() as ss:
        if table.__tablename__ == 'profanity_classes':
            statement = select(ProfanityClasses).where(
                ProfanityClasses.profanity_class == data.profanity_class)
            result = ss.execute(statement).scalars().first()
            ss.close()
            if result:

                return result.id
            else:

                return None

        elif table.__tablename__ == 'semantic_classes':
            statement = select(SemanticClasses).where(
                (SemanticClasses.toxic_class == data.toxic_class) &
                (SemanticClasses.insult_class == data.insult_class) &
                (SemanticClasses.threat_class == data.threat_class) &
                (SemanticClasses.dangerous_class == data.dangerous_class)
            )
            result = ss.execute(statement).scalars().first()
            ss.close()
            if result:

                return result.id
            else:

                return None
        elif table.__tablename__ == 'answers':
            statement = select(Answer).where(
                (Answer.toxic_class == data.toxic_class) &
                (Answer.insult_class == data.insult_class) &
                (Answer.threat_class == data.threat_class) &
                (Answer.dangerous_class == data.dangerous_class) &
                (Answer.profanity_class == data.profanity_class)
            )
            result = ss.execute(statement).scalars().first()
            ss.close()

            if result:

                return result.id
            else:

                return None
        else:
            raise Exception('Incorrect table name')
