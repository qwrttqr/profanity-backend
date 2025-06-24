from db.db_models import (Text, ProfanityClasses,
                          SemanticClasses, Answer)
from datetime import datetime
from db.utils.session import get_session
from db.utils.table_collisions import table_collisions


def collect_information(text_before: str,
                        text_after: str,
                        semantic_classes: dict,
                        profanity_class: int):
    """
    Stores information id db tables.
    Args:
        text_before: str - text before processing
        text_after: str - text after processing
        semantic_classes: dict - list of analyzer classes
        profanity_class: int - profanity class
    """

    LocalSession = get_session()

    with LocalSession() as ss:
        try:
            profanity_class_obj = ProfanityClasses(profanity_class=profanity_class)
            semantic_class_obj = SemanticClasses(
                toxic_class=semantic_classes.get('toxic'),
                insult_class=semantic_classes.get('insult'),
                threat_class=semantic_classes.get('threat'),
                dangerous_class=semantic_classes.get('dangerous'))

            answer_obj = Answer()
            profanity_obj_coll = table_collisions(table=ProfanityClasses,
                                            data=profanity_class_obj)
            semantic_obj_coll = table_collisions(table=SemanticClasses,
                                            data=semantic_class_obj)
            answer_obj_coll = table_collisions(table=Answer,
                                         data=answer_obj)
            if profanity_obj_coll is None:
                profanity_obj_coll = profanity_class_obj
                ss.add(profanity_class_obj)
                ss.flush()

            if semantic_obj_coll is None:
                semantic_obj_coll = semantic_class_obj
                ss.add(semantic_class_obj)
                ss.flush()


            if answer_obj_coll is None:
                answer_obj_coll = answer_obj
                ss.add(answer_obj)
                ss.flush()

            text_obj = Text(text_before_processing=text_before,
                            text_after_processing=text_after,
                            profanity_id=profanity_obj_coll.id,
                            semantic_id=semantic_obj_coll.id,
                            answers_id=answer_obj_coll.id,
                            creation_date=datetime.now(),
                            profanity=profanity_obj_coll,
                            semantic=semantic_obj_coll,
                            answers=answer_obj_coll)
            ss.add(text_obj)
            ss.commit()
        except Exception as e:
            ss.rollback()
            print('Error commiting to db', e)
            raise Exception('Error commiting to db')
