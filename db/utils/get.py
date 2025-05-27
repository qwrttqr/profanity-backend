from db.db_models import Text, ProfanityClasses, SemanticClasses
from db.utils import get_session
from sqlalchemy import select


def get_answers_table_(skip, limit):
    statement = (
        select(
            Text.text_before_processing,
            Text.text_after_processing,
            Text.creation_date,
            ProfanityClasses.profanity_class,
            SemanticClasses.toxic_class,
            SemanticClasses.insult_class,
            SemanticClasses.threat_class,
            SemanticClasses.dangerous_class
        )
        .join(ProfanityClasses, Text.profanity_id == ProfanityClasses.id)
        .join(SemanticClasses, Text.semantic_id == SemanticClasses.id)
    ).offset(skip).limit(limit)

    LocalSession = get_session()
    rows = []
    with LocalSession() as ss:
        result = ss.execute(statement).fetchall()

        for item in result:

            info = {
                'text_before': str(item.text_before_processing),
                'text_after': str(item.text_after_processing),
                'creation_date': item.creation_date.isoformat(),
                'profanity_class': int(item.profanity_class),
                'toxic_class': int(item.toxic_class),
                'insult_class': int(item.insult_class),
                'threat_class': int(item.threat_class),
                'dangerous_class': int(item.dangerous_class)
            }
            rows.append(info)

    return rows