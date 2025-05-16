from sqlalchemy.orm import Mapped
from .base_class import DeclarativeBase
from sqlalchemy.orm import relationship


class profanity_label(DeclarativeBase):
    __tablename__ = 'answers'
    id: Mapped[int] = relationship(back_populates='profanity_id',
                                   primary_key=True, autoincrement=True)
    profanity_class: Mapped[int]

    def __repr__(self) -> str:
        return f'Text(id={self.id!r}, \
                    profanity_class={self.profanity_class!r}'
