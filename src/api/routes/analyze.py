from dawg_python.units import offset
from fastapi import APIRouter
from fastapi import Query
from fastapi import Request
from db.utils import get_answers_table_

router = APIRouter(prefix='/analyze', tags=['profanity'])


@router.get('/')
def analyze_text_get(request: Request,
                     text: str = Query(...,
                                       description='Text to analyze'),
                     threshold: float = 0.5) -> dict[str, str]:
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


@router.get('/answers/')
def get_answers_table(skip: int = Query(default=0,
                                        description='How much rows in table '
                                                    'we should skip'),
                      limit: int = Query(default=20,
                                         description='How much rows we want to get')):
    '''
    Returns path of answers starts from skip+1 row and ends in skip+offset row.
    Parameters:
        skip : int - how much rows to skip.
        limit : int - how much rows we want to get.
    Returns:
        'table_headers': list - headers of the table
        'rows': list - list of array rows
    '''
    table_headers = ['Текст до подготовки',
                     'Текст после подготовки',
                     'Дата обработки',
                     'Содержит маты',
                     'Токсичное',
                     'Содержит оскорбления',
                     'Содержит угрозы',
                     'Содержит репутационный риск для отправителя']

    result = get_answers_table_(skip, limit)

    return {
        'table_headers': table_headers,
        'rows': result
    }
