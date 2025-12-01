import logging
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from functools import lru_cache

from app.db.session import get_db
from app.repositories.vehicle_repository import VehicleRepository
from app.services.vehicle_service import VehicleService
from app.services.vector_service import VectorService
from app.services.embedding_service import EmbeddingService
from app.services.normalization_service import NormalizationService
from app.services.llm_service import LLMService
from app.schemas.vehicle import (
    VehicleRead,
    VehicleCreate,
    VehicleMatchRequest,
    VehicleMatchResponse,
    VehicleBatchMatchRequest,
    VehicleBatchMatchResultFull,
    VehicleBatchMatchResultSimple,
    EmbeddingRequest,
    EmbeddingResponse,
)
from app.repositories.vector_repository import VectorRepository


router = APIRouter()


def get_vehicle_repo(db: Session = Depends(get_db)) -> VehicleRepository:
    return VehicleRepository(db)


def get_vector_repo() -> VectorRepository:
    return VectorRepository()

@lru_cache()
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()

def get_vector_service(
    repo: VectorRepository = Depends(get_vector_repo),
    embedding_service: EmbeddingService = Depends(get_embedding_service)
) -> VectorService:
    return VectorService(repo, embedding_service)

@lru_cache()
def get_normalization_service() -> NormalizationService:
    return NormalizationService()

@lru_cache()
def get_llm_service() -> LLMService:
    return LLMService()

def get_vehicle_service(
    repo: VehicleRepository = Depends(get_vehicle_repo),
    vector_service: VectorService = Depends(get_vector_service),
    normalization_service: NormalizationService = Depends(get_normalization_service),
    llm_service: LLMService = Depends(get_llm_service)
) -> VehicleService:
    return VehicleService(repo, vector_service, normalization_service, llm_service)


@router.get("/vehicles/{crabi_id}", response_model=VehicleRead)
def get_vehicle(crabi_id: str, service: VehicleService = Depends(get_vehicle_service)):
    vehicle = service.get_vehicle_by_crabi_id(crabi_id)
    if not vehicle:
        raise HTTPException(status_code=404, detail="Vehicle not found")
    return vehicle

@router.post("/vehicles/match", response_model=VehicleRead | VehicleMatchResponse | None)
def match_vehicles(request: VehicleMatchRequest, service: VehicleService = Depends(get_vehicle_service)):
    response = service.get_similar_vehicles(request.description, strict=request.strict)
    if not response:
        logging.info(f"No similar vehicles found for description: {request.description}")
        return None
    if request.full_response:
        return response
    return VehicleMatchResponse(id_crabi=response.id_crabi)

@router.post("/vehicles/match/batch", response_model=list[VehicleBatchMatchResultFull] | list[VehicleBatchMatchResultSimple])
def match_vehicles_batch(request: VehicleBatchMatchRequest, service: VehicleService = Depends(get_vehicle_service)):
    results = []
    for description in request.descriptions:
        vehicle = service.get_similar_vehicles(description, strict=request.strict)
        if request.full_response:
            results.append(VehicleBatchMatchResultFull(
                description=description,
                vehicle=vehicle
            ))
        else:
            results.append(VehicleBatchMatchResultSimple(
                description=description,
                id_crabi=vehicle.id_crabi if vehicle else None
            ))
    return results

@router.post("/vehicles", response_model=VehicleRead)
def create_vehicle(vehicle: VehicleCreate, service: VehicleService = Depends(get_vehicle_service)):
    return service.create_vehicle(vehicle.id_crabi, vehicle.description)

@router.post("/vehicles/embedding", response_model=EmbeddingResponse)
def get_embedding(request: EmbeddingRequest, vector_service: VectorService = Depends(get_vector_service)):
    embedding = vector_service.calculate_embedding(request.description)
    return EmbeddingResponse(
        embedding=embedding,
        dimension=len(embedding)
    )
