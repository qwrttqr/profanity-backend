from sqlalchemy.orm import Mapped, relationship
from sqlalchemy.testing.schema import mapped_column

from .base_class import Base


class Answer(Base):
    __tablename__ = 'answers'
    id: Mapped[int] = mapped_column(primary_key=True,
                                    autoincrement=True)
    toxic_class: Mapped[int] = mapped_column(nullable=True)
    insult_class: Mapped[int] = mapped_column(nullable=True)
    threat_class: Mapped[int] = mapped_column(nullable=True)
    dangereous_class: Mapped[int] = mapped_column(nullable=True)
    profanity_class: Mapped[int] = mapped_column(nullable=True)

    texts: Mapped[list['Text']] = relationship(back_populates='answers')

    def __repr__(self) -> str:
        return f'Text(id={self.id!r}, \
                    toxic_class={self.toxic_class!r}, \
                    insuilt_class={self.insult_class!r}, \
                    threat_class={self.threat_class!r}, \
                    dangereous_class={self.dangereous_class!r}, \
                    profanity_class={self.profanity_class!r},'
