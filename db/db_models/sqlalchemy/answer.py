from sqlalchemy.orm import Mapped, mapped_column, relationship
from db.db_models.sqlalchemy.base_class import Base


class Answer(Base):
    __tablename__ = 'answers'
    id: Mapped[int] = mapped_column(primary_key=True,
                                    autoincrement=True)
    toxic_class: Mapped[int] = mapped_column(nullable=True,
                                             default=None)
    insult_class: Mapped[int] = mapped_column(nullable=True,
                                              default = None)
    threat_class: Mapped[int] = mapped_column(nullable=True,
                                              default = None)
    dangerous_class: Mapped[int] = mapped_column(nullable=True,
                                                  default = None)
    profanity_class: Mapped[int] = mapped_column(nullable=True,
                                                 default = None)
    texts: Mapped[list['Text']] = relationship(back_populates='answers')

    def __repr__(self) -> str:
        return f'Answer(id={self.id!r}, \
                toxic_class={self.toxic_class!r},\
                insult_class={self.insult_class!r},\
                threat_class={self.threat_class!r},\
                dangerous_class={self.dangerous_class!r},\
                profanity_class={self.profanity_class!r},'
