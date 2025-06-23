from sqlalchemy import and_
from db.db_models.sqlalchemy import Text


def suitable_row(row, **params):
    for key,value in params.items():
        if value == 'all':
            continue
        else:
            if row[key] != int(value):

                return False

    return True

def filter_rows(rows, **kwargs) -> list:
    filtered_rows = []
    for item in rows:
        if suitable_row(item, **kwargs):
            filtered_rows.append(item)
    return filtered_rows


