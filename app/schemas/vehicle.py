from pydantic import BaseModel


class VehicleRead(BaseModel):
    id: int
    id_crabi: str
    description: str


class VehicleCreate(BaseModel):
    id_crabi: str
    description: str


class VehicleMatchRequest(BaseModel):
    description: str
    full_response: bool = False
    strict: bool = False


class VehicleMatchResponse(BaseModel):
    id_crabi: str


class VehicleBatchMatchRequest(BaseModel):
    descriptions: list[str]
    full_response: bool = False
    strict: bool = False


class VehicleBatchMatchResultFull(BaseModel):
    description: str
    vehicle: VehicleRead | None


class VehicleBatchMatchResultSimple(BaseModel):
    description: str
    id_crabi: str | None


class EmbeddingRequest(BaseModel):
    description: str


class EmbeddingResponse(BaseModel):
    embedding: list[float]
    dimension: int
