from sqlalchemy import String, DateTime, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from db.db_models.base_class import Base
from datetime import datetime


class Text(Base):
    __tablename__ = 'texts'
    id: Mapped[int] = mapped_column(primary_key=True,
                                    autoincrement=True)
    text_before_processing: Mapped[str] = mapped_column(String(4000),
                                                        nullable=False)
    text_after_processing: Mapped[str] = mapped_column(String(4000),
                                                       nullable=False)
    creation_date: Mapped[datetime] = mapped_column(DateTime,
                                                 default=datetime.now,
                                                 nullable=False)

    semantic_id: Mapped[int] = mapped_column(ForeignKey(
        'semantic_classes.id'), nullable=False)
    answers_id: Mapped[int] = mapped_column(ForeignKey('answers.id'),
                                            nullable=False)
    profanity_id: Mapped[int] = mapped_column(ForeignKey(
        'profanity_classes.id'), nullable=False)

    profanity: Mapped['ProfanityClasses'] = relationship(
        back_populates='texts')
    answers: Mapped['Answer'] = relationship(
        back_populates='texts')
    semantic: Mapped['SemanticClasses'] = relationship(
        back_populates='texts')

    def __repr__(self) -> str:
        return f'Text(id={self.id!r}, \
                    semantic_id={self.semantic_id!r}, \
                    profanity_id={self.profanity_id!r}, \
                    answers_id={self.answers_id!r}, \
                    text_before_processing={self.text_before_processing!r}, \
                    text_after_processing={self.text_after_processing!r}, \
                    created_at={self.creation_date!r})'