from db.db_models import Text, ProfanityClasses, SemanticClasses, Answer
from sqlalchemy import select

select_from_model_answers_statement = (
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
)

select_from_answers_statement = (
    select(
        Text.id,
        Text.text_before_processing,
        Text.text_after_processing,
        ProfanityClasses.profanity_class,
        SemanticClasses.toxic_class,
        SemanticClasses.insult_class,
        SemanticClasses.threat_class,
        SemanticClasses.dangerous_class
    )
    .join(ProfanityClasses, Text.profanity_id == ProfanityClasses.id)
    .join(SemanticClasses, Text.semantic_id == SemanticClasses.id)
)