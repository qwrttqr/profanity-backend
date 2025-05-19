from db.db_models import (Text, ProfanityClasses,
                          SemanticClasses, Answer)
from datetime import datetime
from db.utils.session import get_session
from db.utils.table_collisions import table_collisions


def collect_information(text_before: str,
                        text_after: str,
                        analyzer_classes: dict,
                        profanity_class: int):
    '''
    Collects information from analyzer and manages it to store to db.
    :param text_before: str - text before processing
    :param text_after: str - text after processing
    :param analyzer_classes: dict - list of analyzer classes
    :param profanity_class: int - profanity class
    '''

    LocalSession = get_session()

    with LocalSession() as ss:
        try:
            profanity_class_obj = ProfanityClasses(
                profanity_class=profanity_class)
            semantic_class_obj = SemanticClasses(
                toxic_class=analyzer_classes.get('toxic'),
                insult_class=analyzer_classes.get('insult'),
                threat_class=analyzer_classes.get('threat'),
                dangerous_class=analyzer_classes.get('dangerous'))

            answer_obj = Answer()
            profanity_id = table_collisions(table=ProfanityClasses,
                                            data=profanity_class_obj)
            if profanity_id is None:
                ss.add(profanity_class_obj)
                ss.flush()
                profanity_id = profanity_class_obj.id

            semantic_id = table_collisions(table=SemanticClasses,
                                           data=semantic_class_obj)
            if semantic_id is None:
                ss.add(semantic_class_obj)
                ss.flush()
                semantic_id = semantic_class_obj.id

            ss.add(answer_obj)
            ss.flush()
            answer_id = table_collisions(table=Answer,
                                         data=answer_obj)

            if answer_id is None:
                ss.add(answer_obj)
                ss.flush()
                answer_id = answer_obj.id

            text_obj = Text(text_before_processing=text_before,
                            text_after_processing=text_after,
                            profanity_id=profanity_id,
                            semantic_id=semantic_id,
                            answers_id=answer_id,
                            creation_date=datetime.now())
            ss.add(text_obj)
            ss.commit()
        except Exception as e:
            ss.rollback()
            print('error commiting to db', e)
