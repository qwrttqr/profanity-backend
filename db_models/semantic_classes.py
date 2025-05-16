from sqlalchemy.orm import Mapped
from sqlalchemy.testing.schema import mapped_column

from .base_class import Base
from sqlalchemy.orm import relationship


class SemanticClasses(Base):
    __tablename__ = 'semantic_classes'
    id: Mapped[int] = mapped_column(primary_key=True,
                                    autoincrement=True)
    toxic_class: Mapped[int] = mapped_column(nullable=False)
    insult_class: Mapped[int] = mapped_column(nullable=False)
    threat_class: Mapped[int] = mapped_column(nullable=False)
    dangereous_class: Mapped[int] = mapped_column(nullable=False)

    texts: Mapped[list['Text']] = relationship(back_populates='semantic')

    def __repr__(self) -> str:
        return f'Text(id={self.id!r}, \
                    toxic_class={self.toxic_class!r}, \
                    insuilt_class={self.insult_class!r}, \
                    threat_class={self.threat_class!r}, \
                    dangereous_class={self.dangereous_class!r},'
