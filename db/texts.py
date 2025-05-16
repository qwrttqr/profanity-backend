from .init_db import Session
from db_models.text import Text

def add_to_texts(profanity_id, semantic_id, answers_id, text_before, text_after):

    with Session as session:
        new_row = Text(

        )
