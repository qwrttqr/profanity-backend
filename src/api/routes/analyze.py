from fastapi import APIRouter
from fastapi import Query
from fastapi import Request

router = APIRouter(prefix='/analyze', tags=['profanity'])


@router.get('/')
def analyze_text_get(request: Request
                     ,text: str = Query(...,
                                        description='Text to analyze'),
                     threshold: float = 0.12) -> dict[str, str]:
    '''
    Analyze text toxicity via GET request.
    Parameters:
        text: str - text, that should be analyzed.
        threshold: float - threshold for mark text as a toxic.
        request: Request - request
    Returns:
        text: text which was analyzed, label: toxicity label.
        profanity: do text has or not profanity words.
    '''
    print(text)
    labels = request.app.state.analyzer.analyze(text, threshold)
    class_ = labels['text_labels']['toxic']
    has_profanity = labels['profanity_label']

    return {
        'text': text,
        'label': 'Текст не ок' if class_ == 1 else 'Текст ок',
        'profanity': 'Есть маты' if has_profanity else 'Матов нет'
    }
