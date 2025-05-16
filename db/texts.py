from db.init_db import db_engine
from sqlalchemy.orm import sessionmaker
from db_models import Text


def save_to_texts(**kwargs):
        text_before = kwargs['text_before']
        text_after = kwargs['text_after']
        created_at = kwargs['created_at']
        semantic_id = kwargs['semantic_id']
        answers_id = kwargs['answers_id']
        profanity_id = kwargs['profanity_id']



        new_text = Text(text_before_processing=text_before,
                        text_after_processing=text_after,
                        created_at=created_at,
                        semantic_id=semantic_id,
                        answers_id=answers_id,
                        profanity_id=profanity_id
                        )

        session = sessionmaker(db_engine)
        with session() as ss:
            ss.add(new_text)
            ss.commit()