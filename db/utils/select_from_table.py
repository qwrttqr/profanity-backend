import datetime

from dawg_python.units import offset

from db.utils import get_session
from sqlalchemy import Select


def select_from_table(statement: Select, skip: int = -1, limit: int = -1):
    """
    Executes select statement by given statement.
    Args:
        statement: Select - select statement.
        skip: int - how many rows to skip for pagination.
        limit: int - how many rows to select after skipped ones

    Returns:
        rows: list[dict] - list of rows.

    Raises:
        Error during selecting from table.
    """
    LocalSession = get_session()
    rows = []
    with LocalSession() as ss:
        try:
            if skip > -1:
                statement = statement.offset(skip)
            if limit > -1:
                statement = statement.limit(limit)
            res = ss.execute(statement).fetchall()
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

        ss.close()

    return rows
