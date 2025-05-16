from fastapi import APIRouter
from fastapi import Query
from src.utils import text_analyzer

router = APIRouter(prefix='/analyze', tags=['profanity'])


@router.get('/')
def analyze_text_get(text: str = Query(..., description='Text to analyze'),
                     threshold: float = 0.5) -> dict[str, str]:
    '''
    Analyze text toxicity via GET request.
    Parameters:
        text: str - text, that should be analyzed.
        threshold: float - threshold for mark text as a toxic.
    Returns:
        text: text which was analyzed, label: toxicity label.
        profanity: do text has or not profanity words.
    '''
    print(text)
    # labels = text_analyzer.analyze(text, threshold)
    # class_ = labels['text_labels']['toxic']
    # has_profanity = labels['profanity_label']

    return {
        'text': text,
        'label': 'Текст не ок' if class_ == 1 else 'Текст ок',
        'profanity': 'Есть маты' if has_profanity else 'Матов нет'
    }
