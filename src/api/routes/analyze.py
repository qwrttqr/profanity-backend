from fastapi import APIRouter
from fastapi import Query
from src.utils.text_analyzer import text_analyzer

router = APIRouter(prefix='/analyze', tags=['profanity'])


@router.get('/')
def analyze_text_get(text: str = Query(..., description="Text to analyze"), threshold: float = 0.12) -> dict[str, str]:
    """
    Analyze text toxicity via GET request.
    Parameters:
        text: str - text, that should be analyzed.
        threshold: float - threshold for mark text as a toxic.
    Returns:
        text: text which was analyzed, label: toxicity label, profanity: do text has or not profanity words. 
    """
    
    class_ = text_analyzer.analyze_toxicity(
        text, return_proba=False, threshold=threshold)
    
    has_profanity = any(i for i in text_analyzer.predict_profanity(text))

    return {
        "text": text,
        "label": 'Текст не ок' if class_ == 1 else 'Текст ок',
        "profanity": "Есть маты" if has_profanity else "Матов нет"
    }
