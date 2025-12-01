from sqlalchemy.orm import Session

from app.models.vehicle import Vehicle
from app.repositories.base import BaseRepository


class VehicleRepository(BaseRepository[Vehicle]):
    def __init__(self, db: Session):
        super().__init__(Vehicle, db)

    def get_by_crabi_id(self, crabi_id: str) -> Vehicle | None:
        return self.db.query(self.model).filter(self.model.id_crabi == crabi_id).first()

    def get_by_crabi_ids(self, crabi_ids: list[str]) -> list[Vehicle]:
        """
        Retrieve multiple vehicles by their crabi_ids.
        
        Args:
            crabi_ids: List of crabi_id strings to look up
            
        Returns:
            List of Vehicle objects found (may be fewer than requested if some don't exist)
        """
        if not crabi_ids:
            return []
        return self.db.query(self.model).filter(self.model.id_crabi.in_(crabi_ids)).all()

    def create_vehicle(self, vehicle: Vehicle) -> Vehicle:
        self.db.add(vehicle)
        self.db.commit()
        self.db.refresh(vehicle)
        return vehicle