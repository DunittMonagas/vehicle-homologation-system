
from typing import Any, Generic, Type, TypeVar

from sqlalchemy.orm import Session

from app.models.base import Base


ModelType = TypeVar("ModelType", bound=Base)

class BaseRepository(Generic[ModelType]):
    def __init__(self, model: Type[ModelType], db: Session):
        self.model = model
        self.db = db

    def get(self, id: Any) -> ModelType | None:
        return self.db.query(self.model).filter(self.model.id == id).first()
