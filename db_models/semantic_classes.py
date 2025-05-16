from sqlalchemy.orm import Mapped
from .base_class import DeclarativeBase
from sqlalchemy.orm import relationship


class semantic_label(DeclarativeBase):
    __tablename__ = 'semantic_classes'
    id: Mapped[int] = relationship(back_populates='semantic_id',
                                   primary_key=True, autoincrement=True)
    toxic_class: Mapped[int]
    insult_class: Mapped[int]
    threat_class: Mapped[int]
    dangereous_class: Mapped[int]

    def __repr__(self) -> str:
        return f'Text(id={self.id!r}, \
                    toxic_class={self.toxic_class!r}, \
                    insuilt_class={self.insult_class!r}, \
                    threat_class={self.threat_class!r}, \
                    dangereous_class={self.dangereous_class!r},'
