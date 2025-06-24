def suitable_row(row, **params):
    for key,value in params.items():
        if value == 'all':
            continue
        else:
            if row[key] != int(value):

                return False

    return True

def build_where_clause(rows, **kwargs) -> list:
    for

