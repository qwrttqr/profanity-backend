import json
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
    with LocalSession() as ss:
        rows = []
        result = ss.execute(statement).fetchall()
        for item in result:
            rows.append(item._asdict())
        return json.dumps(result)
        ss.close()