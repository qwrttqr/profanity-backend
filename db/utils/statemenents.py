from db.db_models import Text, ProfanityClasses, SemanticClasses
from sqlalchemy import select, update

########################## SELECT STATEMENTS ##########################


# Select table to show on backend
select_table = (
    select(
        Text.id,
        Text.text_before_processing,
        Text.text_after_processing,
        Text.creation_date,
        Text.updation_date,
        ProfanityClasses.profanity_class,
        SemanticClasses.toxic_class,
        SemanticClasses.insult_class,
        SemanticClasses.threat_class,
        SemanticClasses.dangerous_class
    )
    .join(ProfanityClasses, Text.profanity_id == ProfanityClasses.id)
    .join(SemanticClasses, Text.semantic_id == SemanticClasses.id)
)

# Select for profanity post-learning
select_from_model_answers_for_profanity = (
    select(
        Text.id,
        Text.text_after_processing,
        ProfanityClasses.profanity_class
    ).join(Text.profanity)
)

# Select for semantic post-learning
select_from_model_answers_for_semantic = (
    select(
        Text.id,
        Text.text_after_processing,
        SemanticClasses.toxic_class,
        SemanticClasses.insult_class,
        SemanticClasses.threat_class,
        SemanticClasses.dangerous_class
    ).join(Text.semantic)
)


########################## UPDATE STATEMENTS ##########################


# Update model answers table
update_profanity_id = (
    update(Text)
)

update_semantic_id = (
    update(SemanticClasses)
)