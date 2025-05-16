from db import (save_to_semantic_classes, save_to_texts,
                save_to_profanity_classes, save_to_answer)
from datetime import datetime

def collect_information(text_before: str,
                        text_after: str,
                        analyzer_classes: list,
                        profanity_class: int):
    '''
    Collects information from analyzer and manages it to store to db.
    :param text_before: str - text before processing
    :param text_after: str - text after processing
    :param analyzer_classes: list - list of analyzer classes
    :param profanity_class: int - profanity class
    '''

    profanity_id = save_to_profanity_classes(profanity_class)
    semantic_id = save_to_semantic_classes(analyzer_classes)
    answred_id = save_to_answer(analyzer_classes)
    information = {
        'text_before_processing': text_before,
        'text_after_processing': text_after,
        'profanity_id': profanity_id,
        'semantic_id': semantic_id,
        'text': text_before,
        'created_at': datetime.now(),
        'anwser_id': answred_id
    }
    save_to_texts(information)
