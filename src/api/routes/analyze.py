from sqlalchemy import or_
from fastapi import APIRouter, Query, Request, HTTPException
from db.utils import select_from_table, build_where_clauses
from db.utils.statemenents import select_table
from db.db_models.pydantic import AnswerPost
from db.db_models.sqlalchemy.text import Text
from src.utils.file_work import files
from src.utils.post_learn.splitter import split

router = APIRouter(prefix='/analyze', tags=['model interaction'])


@router.get('/', status_code=200)
def analyze_text_get(request: Request,
                     text: str = Query(...,
                                       description='Text to analyze'),
                     threshold: float = 0.5) -> dict[str, str]:
    """
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
    """
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


@router.get('/table')
def get_model_answers_table(skip: int = Query(default=0,
                                              description='How much rows in table '
                                                          'we should skip'),
                            limit: int = Query(default=20,
                                               description='How much rows we want to get'),
                            profanity_class: str = Query(default='all',
                                                         description='Filter for profanity class field'),
                            toxic_class: str = Query(default='all',
                                                     description='Filter for toxic class field'),
                            insult_class: str = Query(default='all',
                                                      description='Filter for insult class field'),
                            threat_class: str = Query(default='all',
                                                      description='Filter for threat class field'),
                            dangerous_class: str = Query(default='all',
                                                         description='Filter for dangerous class field')):
    """

    Returns rows of model answers starts from skip+1 row and ends in
    skip+offset row.
    Parameters:
        dangerous_class: str - filter for dangerous class field
        threat_class: str - filter for threat class field
        insult_class: str - filter for insult class field
        toxic_class: str - filter for toxic class field
        profanity_class: str - filter for profanity class field
        skip: int - how much rows to skip.
        limit: int - how much rows we want to get.
    Returns:
        'table_headers': list - headers of the table
        'rows': list - list of array rows
    """

    where_clauses = build_where_clauses(profanity_class=profanity_class,
                                        toxic_class=toxic_class,
                                        insult_class=insult_class,
                                        threat_class=threat_class,
                                        dangerous_class=dangerous_class)

    result = select_from_table(select_table, skip, limit, where_clauses)

    table_headers = [{'text': 'Текст до', 'filterable': False},
                     {'text': 'Текст после', 'filterable': False},
                     {'text': 'Дата обработки', 'filterable': False},
                     {'text': 'Дата обновления классов', 'filterable': False},
                     {'text': 'Содержит маты', 'filterable': True, 'key': 'profanity_class',
                      'options': ['Все', '1', '0']},
                     {'text': 'Токсичное', 'filterable': True, 'key': 'toxic_class',
                      'options': ['Все', '1', '0']},
                     {'text': 'Содержит оскорбления', 'filterable': True,
                      'key': 'insult_class', 'options': ['Все', '1', '0']},
                     {'text': 'Содержит угрозы', 'filterable': True, 'key': 'threat_class',
                      'options': ['Все', '1', '0']},
                     {'text': 'Содержит репутационный риск для отправителя', 'filterable': True,
                      'key': 'dangerous_class', 'options': ['Все', '1', '0']}]
    return {
        'rows': result,
        'headers': table_headers
    }


@router.post('/upload_answers_for_metrics', status_code=200)
def upload_answers_for_metrics(request: Request,
                     answers: AnswerPost,
                     threshold: float = 0.5):
    """
    Accepts list of rows with data for post-learning.
    Args:
        request: Request object
        answers: list - edited rows with actual answers
        threshold: float(optional) - threshold by exceeding which label in classification will be 1
    Returns:
        updated rows: list - list of rows that was processed by new model instances
    """
    profanity_module = request.app.state.profanity_module
    semantic_module = request.app.state.semantic_module
    text_analyzer = request.app.state.analyzer
    profane_rows, semantic_rows = split(answers.rows)
    metrics = []

    try:
        if profane_rows:
            metrics_obj = {
                # Header to show on frontend
                'name': 'Метрики profanity модели',
                # Metrics type
                'type': 'profanity'
            }
            # Collect all the metrics
            for key, value in (profanity_module.post_learn(profanity_rows=profane_rows,
                                                           text_analyzer=text_analyzer)).items():
                metrics_obj[key] = value
            metrics.append(metrics_obj)
        if semantic_rows:
            metrics_obj = {
                # Header to show on frontend
                'name': 'Метрики semantic модели',
                # Metrics type
                'type': 'semantic'
            }
            # Collect all the metrics
            for key, value in (semantic_module.post_learn(semantic_rows=semantic_rows,
                                                           text_analyzer=text_analyzer,
                                                           threshold=threshold)).items():
                metrics_obj[key] = value
            metrics.append(metrics_obj)
        return {'metrics': metrics}

    except Exception as e:
        print(f'Error during post-learning: {str(e)}')
        raise HTTPException(status_code=500,
                            detail=f'Error during post-learning: {str(e)}')


@router.post('/update_profanity_model', status_code=200)
def update_profanity_model(request: Request, profanity_rows: AnswerPost) -> dict[str, list]:
    """
       Post-learns profanity model.
       Args:
           request: Request obj\n
           profanity_rows: list - list of rows with new profanity answers\n

       Returns:
           updated rows: dict[str, list] - list of rows that was processed by new model instances
       """
    profanity_module = request.app.state.profanity_module
    text_analyzer = request.app.state.analyzer
    try:
        profanity_module.post_learn(profanity_rows=profanity_rows.rows,
                                    text_analyzer=text_analyzer,
                                    save_model=True)

        where_clauses = [or_(*[Text.id == item['id'] for item in profanity_rows.rows])]

        result = select_from_table(select_table, where_clauses=where_clauses)
        return {'updated_rows': result}
    except Exception as e:
        print(f'Error during post-learning: {str(e)}')
        raise HTTPException(status_code=500,
                            detail=f'Error during post-learning: {str(e)}')

@router.post('/update_semantic_model', status_code=200)
def update_sematic_model(request: Request, semantic_rows: AnswerPost,  threshold: float = 0.5) -> dict[str, list]:
    """
    Post-learns semantic model.
    Args:
        request: Request obj\n
        semantic_rows: list - list of rows with new semantic answers\n
        threshold: float(optional) - threshold by exceeding which label in classification will be 1

    Returns:
        updated rows: dict[str, list] - list of rows that was processed by new model instances
    """
    text_analyzer = request.app.state.analyzer
    semantic_module = request.app.state.semantic_module
    try:
        semantic_module.post_learn(semantic_rows=semantic_rows.rows,
                                   text_analyzer=text_analyzer,
                                   threshold=threshold,
                                   save_model=True)

        where_clauses = [or_(*[Text.id == item['id'] for item in semantic_rows.rows])]

        result = select_from_table(select_table, where_clauses=where_clauses)
        return {'updated_rows': result}
    except Exception as e:
        print(f'Error during post-learning: {str(e)}')
        raise HTTPException(status_code=500,
                            detail=f'Error during post-learning: {str(e)}')


@router.get('/get_models_info', status_code=200)
def get_models_info():
    try:

        return {'profanity_model_info': files['profanity_model_info'],
                'semantic_model_info': files['semantic_model_info']}
    except Exception as e:
        print(f'Error returning model info objects: {str(e)}')

        raise HTTPException(status_code=500,
                            detail=f'Error returning model info objects: {str(e)}')
