from datetime import datetime
from db.db_models.sqlalchemy.text import Text
from db.utils import get_session
from sqlalchemy import Update
from db.utils.statemenents import update_profanity_id
from .table_collisions import table_collisions
from db.db_models.sqlalchemy.profanity_classes import ProfanityClasses
from db.db_models.sqlalchemy.semantic_classes import SemanticClasses

def get_id(table_type: str,
           semantic_classes: dict[str, int] = None, profanity_class: int = None) -> int | None:
    """
    Trying to find id with same classes or creates new table row
    Args:
        table_type: str Table - table that where we will find(semantic_table or profanity_table)
        semantic_classes: dict - optional, dictionary with semantic classes, default None
        profanity_class: int - optional, integer contains profanity class, default None
    Returns:
        row_id: int - id for table row
    """
    LocalSession = get_session()

    with LocalSession() as ss:
        if table_type == 'profanity_table':
            profanity_class_obj = ProfanityClasses(
                profanity_class=profanity_class)
            profanity_id = table_collisions(table=ProfanityClasses,
                                            data=profanity_class_obj)
            if profanity_id is None:
                ss.add(profanity_class_obj)
                ss.flush()
                ss.commit()

                return profanity_class_obj.id

            return profanity_id
        elif table_type == 'semantic_table':
            semantic_class_obj = SemanticClasses(
                toxic_class=semantic_classes.get('toxic'),
                insult_class=semantic_classes.get('insult'),
                threat_class=semantic_classes.get('threat'),
                dangerous_class=semantic_classes.get('dangerous'))

            semantic_id = table_collisions(table=SemanticClasses,
                                           data=semantic_class_obj)
            if semantic_id is None:
                ss.add(semantic_class_obj)
                ss.flush()
                ss.commit()

                return semantic_class_obj.id

            return semantic_id
        else:
            ss.rollback()
            raise Exception('Error during id lookup, wrong table')

        return None



def update_table(statement: Update,
                 where: dict[str, ...],
                 values: dict[str, ...],
                 add_timestamp: bool = True):
    """
    Executes update statement by given statement and id.
    Args:
        statement: Update - update statement
        where: dict[str, ...] - where clause
        values: dict[str, ...] - which values to update
        add_timestamp: bool - optional, True by default

    Raises:
        Error during updating from table.
    """
    if add_timestamp:
        if hasattr(Text, 'updation_date'):
            values['updation_date'] = datetime.now()

    LocalSession = get_session()

    where_clauses = []
    for field_name, value in where.items():
        field = getattr(Text, field_name)
        where_clauses.append(field == value)

    with LocalSession() as ss:
        try:
            stmt = update_profanity_id.where(*where_clauses).values(**values)
            ss.execute(stmt)
            ss.commit()
        except Exception as e:
            print('Error during updating', e)


