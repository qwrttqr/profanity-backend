import datetime
from db.utils import get_session
from sqlalchemy import Select


def select_from_table(statement: Select, skip: int = -1, limit: int = -1):
    '''
    Executes select statement by given statement.
    Args:
        statement: Select - select statement.
        skip: int - how many rows to skip for pagination.
        limit: int - how nany rows to select after skippend ones

    Returns:
        rows: list[dict] - list of rows.

    Raises:
        Error during selecting from table.
    '''
    LocalSession = get_session()
    rows = []
    with LocalSession() as ss:
        try:
            res = ss.execute(statement.offset(skip).limit(limit)).fetchall()
            for item in res:
                row = {}
                for key, value in item._asdict().items():
                    if isinstance(value, datetime.date):
                        row[key] = value.isoformat()
                    else:
                        row[key] = value
                rows.append(row)
        except Exception as e:
            print('Error during selecting from table', e)

    return rows
