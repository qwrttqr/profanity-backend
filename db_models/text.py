from sqlalchemy import String, DateTime
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from .base_class import DeclarativeBase
from datetime import datetime
from sqlalchemy.orm import relationship


class Text(DeclarativeBase):
    __tablename__ = 'texts'
    id: Mapped[int] = mapped_column(primary_key=True, auto_increment=True)
    profanity_id: Mapped[int] = relationship(backbone_populates='id')
    semantic_id: Mapped[int] = relationship(backbone_populates='id')
    answers_id: Mapped[int] = relationship(backbone_populates='id')
    text_before_processing: Mapped[str] = mapped_column(String(4000))
    text_after_processing: Mapped[str] = mapped_column(String(4000))

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)

    def __repr__(self) -> str:
        return f'Text(id={self.id!r}, \
                    semantic_id={self.semantic_id!r}, \
                    profanity_id={self.profanity_id!r}, \
                    answers_id={self.answers_id!r}, \
                    text_before_processing={self.text_before_processing!r}, \
                    text_after_processing={self.text_after_processing!r}, \
                    created_at={self.created_at!r})'
