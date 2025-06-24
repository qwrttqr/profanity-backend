from sqlalchemy import and_
from sqlalchemy.sql import ColumnElement
from db.db_models import Text, SemanticClasses, ProfanityClasses


def get_arr(val):
    if val == 'all':
        return [1, 0]
    else:
        return [int(val)]


def build_where_clauses(**kwargs) -> list[ColumnElement]:
    where_clauses = []

    for key, value in kwargs.items():
        if key == 'profanity_class':
            where_clauses.append(
                Text.profanity.has(ProfanityClasses.profanity_class.in_(get_arr(value)))
            )

        elif hasattr(SemanticClasses, key):
            attr = getattr(SemanticClasses, key)
            where_clauses.append(
                Text.semantic.has(attr.in_(get_arr(value)))
            )

        else:
            raise ValueError(f"Unsupported filter key: {key}")

    return [and_(*where_clauses)]
