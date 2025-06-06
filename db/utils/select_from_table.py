import datetime

from dawg_python.units import offset

from db.utils import get_session
from sqlalchemy import Select


def select_from_table(statement: Select,
                      skip: int = -1, limit: int = -1,
                      where_clauses = None):
    """
    Executes a generic SELECT statement with optional WHERE, ORDER BY, SKIP, LIMIT.

    Args:
        statement (Select): Base SQLAlchemy Select object
        where_clauses (list): List of SQLAlchemy filter expressions (e.g., User.age > 30)
        skip (int): Number of records to skip
        limit (int): Max number of records to return

    Returns:
        list[dict]: List of rows as dictionaries, date fields are ISO formatted
    """
    LocalSession = get_session()
    rows = []
    with LocalSession() as ss:
        try:
            if skip > -1:
                statement = statement.offset(skip)
            if limit > -1:
                statement = statement.limit(limit)
            if where_clauses:
                statement = statement.where(*where_clauses)
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
            print(f'Error during selecting from table by statement {statement}, {str(e)}')
            raise

        ss.close()

    return rows
