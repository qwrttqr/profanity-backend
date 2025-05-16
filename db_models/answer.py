from sqlalchemy.orm import Mapped, relationship
from sqlalchemy.testing.schema import mapped_column

from .base_class import DeclarativeBase


class answer(DeclarativeBase):
    __tablename__ = 'answers'
    id: Mapped[int] = mapped_column(primary_key=True, auto_increment=True)
    text_id: Mapped[int] = relationship(back_populates='answers_id')
    toxic_class: Mapped[int]
    insult_class: Mapped[int]
    threat_class: Mapped[int]
    dangereous_class: Mapped[int]
    profanity_class: Mapped[int]

    def __repr__(self) -> str:
        return f'Text(id={self.id!r}, \
                    toxic_class={self.toxic_class!r}, \
                    insuilt_class={self.insult_class!r}, \
                    threat_class={self.threat_class!r}, \
                    dangereous_class={self.dangereous_class!r}, \
                    profanity_class={self.profanity_class!r},'
