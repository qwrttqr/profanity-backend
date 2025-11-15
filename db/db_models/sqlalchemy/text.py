from sqlalchemy import String, DateTime, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from db.db_models.sqlalchemy.base_class import Base
from datetime import datetime


class Text(Base):
    __tablename__ = 'texts'
    id: Mapped[int] = mapped_column(primary_key=True,
                                    autoincrement=True)
    text_before_processing: Mapped[str] = mapped_column(String(50000),
                                                        nullable=False)
    text_after_processing: Mapped[str] = mapped_column(String(50000),
                                                       nullable=False)
    creation_date: Mapped[datetime] = mapped_column(DateTime,
                                                 default=datetime.now,
                                                 nullable=False)
    updation_date: Mapped[datetime] = mapped_column(DateTime, default=None)

    semantic_id: Mapped[int] = mapped_column(ForeignKey(
        'semantic_classes.id'), nullable=False)
    answers_id: Mapped[int] = mapped_column(ForeignKey(
        'answers.id'), nullable=False)
    profanity_id: Mapped[int] = mapped_column(ForeignKey(
        'profanity_classes.id'), nullable=False)

    profanity: Mapped['ProfanityClasses'] = relationship(
        back_populates='texts')
    answers: Mapped['Answer'] = relationship(
        back_populates='texts')
    semantic: Mapped['SemanticClasses'] = relationship(
        back_populates='texts')

    def __repr__(self) -> str:
        return (
            f"Text(id={self.id!r}, "
            f"text_before_processing={self.text_before_processing!r}, "
            f"text_after_processing={self.text_after_processing!r}, "
            f"creation_date={self.creation_date!r}, "
            f"updation_date={self.updation_date!r}, "
            f"semantic_id={self.semantic_id!r}, "
            f"answers_id={self.answers_id!r}, "
            f"profanity_id={self.profanity_id!r}, "
            f"profanity={self.profanity!r}, "
            f"answers={self.answers!r}, "
            f"semantic={self.semantic!r})"
        )