from sqlalchemy.orm import Mapped, relationship
from sqlalchemy.testing.schema import mapped_column

from .base_class import Base


class ProfanityClasses(Base):
    __tablename__ = 'profanity_classes'
    id: Mapped[int] = mapped_column(primary_key=True,
                                    autoincrement=True)
    profanity_class: Mapped[int] = mapped_column(nullable=False)

    texts: Mapped[list['Text']] = relationship(back_populates="profanity")

    def __repr__(self) -> str:
        return f'Text(id={self.id!r}, \
                    profanity_class={self.profanity_class!r}'
