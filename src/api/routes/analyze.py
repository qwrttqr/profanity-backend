from fastapi import (APIRouter, Query, Request, HTTPException)
from db.utils import select_from_table
from db.utils import select_from_answers_statement, select_from_model_answers_statement
from db.db_models.pydantic import AnswerPost
from src.utils.profanity_module import ProfanityModule
from src.utils.post_learn.splitter import split

router = APIRouter(prefix='/analyze', tags=['profanity'])

@router.get('/', status_code=200)
def analyze_text_get(request: Request,
                     text: str = Query(...,
                                       description='Text to analyze'),
                     threshold: float = 0.5) -> dict[str, str]:
    '''
    Analyze text toxicity via GET request.
    Args:
        text: str - text, that should be analyzed.
        threshold: float - threshold for mark text as a toxic.
        request: Request - request
    Returns:
        text: text which was analyzed, label: toxicity label.
        profanity: do text has or not profanity words.
    Raises:
        HTTP Exception - 400 status code if text for analyzing is incorrect.
    '''
    try:
        labels = request.app.state.analyzer.analyze(text, threshold)
        class_ = labels['text_labels']['toxic']
        has_profanity = labels['profanity_label']

        return {
            'text': text,
            'label': 'Текст не ок' if class_ else 'Текст ок',
            'profanity': 'Есть маты' if has_profanity else 'Матов нет'
        }
    except Exception as e:
        raise HTTPException(status_code=400,
                            detail=f'Invalid input or analysis error: {str(e)}')


@router.get('/model_answers/')
def get_model_answers_table(skip: int = Query(default=0,
                                        description='How much rows in table '
                                                    'we should skip'),
                      limit: int = Query(default=20,
                                         description='How much rows we want to get')):
    '''
    Returns rows of model answers starts from skip+1 row and ends in
    skip+offset row.
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

    result = select_from_table(select_from_model_answers_statement, skip, limit)

    return {
        'table_headers': table_headers,
        'rows': result
    }


@router.get('/answers/')
def get_answers_table(skip: int = Query(default=0,
                                        description='How much rows in table '
                                                    'we should skip'),
                      limit: int = Query(default=20,
                                         description='How much rows we want to get')):
    '''
    Returns rows of answers starts from skip+1 row and ends in skip+offset row.
    Parameters:
        skip : int - how much rows to skip.
        limit : int - how much rows we want to get.
    Returns:
        'table_headers': list - headers of the table
        'rows': list - list of array rows
    '''
    table_headers = ['Текст до подготовки',
                     'Текст после подготовки',
                     'Содержит маты',
                     'Токсичное',
                     'Содержит оскорбления',
                     'Содержит угрозы',
                     'Содержит репутационный риск для отправителя']

    result = select_from_table(select_from_answers_statement, skip, limit)
    return {
        'table_headers': table_headers,
        'rows': result
    }

@router.post('/upload_answers/')
def load_new_answers(request: Request,
                     answers: AnswerPost):
    profanity_module = request.app.state.profanity_module
    profane_rows, semantic_rows = split(answers.rows)
    profanity_module.post_learn(profanity_rows=profane_rows)