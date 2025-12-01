from sqlalchemy import String

from sqlalchemy.orm import Mapped, mapped_column
from app.models.base import Base

class Vehicle(Base):
    __tablename__ = "vehicle"

    id: Mapped[int] = mapped_column(primary_key=True)
    id_crabi: Mapped[str] = mapped_column(String, index=True, unique=True)
    description: Mapped[str] = mapped_column(String, index=False)