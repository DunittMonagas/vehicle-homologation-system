from contextlib import asynccontextmanager
from fastapi import FastAPI
import logging

from app.api.v1 import vehicle
from app.core.config import config
from app.core.logging import setup_logging
from app.db.session import engine
from app.models.base import Base
from app.api.v1.vehicle import get_embedding_service

setup_logging()
logger = logging.getLogger(__name__)

Base.metadata.create_all(bind=engine)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load the embedding model
    logger.info("Loading embedding model...")
    get_embedding_service()
    logger.info("Embedding model loaded successfully.")
    yield
    # Shutdown logic (if any) can go here

app = FastAPI(title=config.app_name, lifespan=lifespan)


# Register routes
app.include_router(vehicle.router, prefix="/api/v1")
