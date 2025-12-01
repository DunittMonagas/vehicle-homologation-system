#!/usr/bin/env python3
"""
Database Population Script

This script reads a CSV file and populates:
1. PostgreSQL database with vehicle records
2. Upstash Vector database with vehicle embeddings

Usage:
    python populate.py --csv /path/to/vehicles.csv
"""

import argparse
import csv
import logging
import os
import sys
from dataclasses import dataclass
from typing import Any

import requests
from langchain_huggingface import HuggingFaceEmbeddings
from sqlalchemy import create_engine, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, Session, sessionmaker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration (using same environment variable names as the main application)
# ============================================================================

@dataclass
class Config:
    """Configuration loaded from environment variables."""
    db_user: str
    db_password: str
    db_name: str
    db_host: str
    db_port: int
    upstash_vector_rest_url: str
    upstash_vector_rest_token: str
    embedding_model: str
    
    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables."""
        return cls(
            db_user=os.environ.get("DB_USER", ""),
            db_password=os.environ.get("DB_PASSWORD", ""),
            db_name=os.environ.get("DB_NAME", ""),
            db_host=os.environ.get("DB_HOST", "localhost"),
            db_port=int(os.environ.get("DB_PORT", "5432")),
            upstash_vector_rest_url=os.environ.get("UPSTASH_VECTOR_REST_URL", ""),
            upstash_vector_rest_token=os.environ.get("UPSTASH_VECTOR_REST_TOKEN", ""),
            embedding_model=os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        )
    
    @property
    def db_url(self) -> str:
        """Build PostgreSQL connection URL."""
        return f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"
    
    def validate(self) -> None:
        """Validate required configuration values."""
        missing = []
        if not self.db_user:
            missing.append("DB_USER")
        if not self.db_password:
            missing.append("DB_PASSWORD")
        if not self.db_name:
            missing.append("DB_NAME")
        if not self.upstash_vector_rest_url:
            missing.append("UPSTASH_VECTOR_REST_URL")
        if not self.upstash_vector_rest_token:
            missing.append("UPSTASH_VECTOR_REST_TOKEN")
        
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")


# ============================================================================
# Database Models (matching main application)
# ============================================================================

class Base(DeclarativeBase):
    pass


class Vehicle(Base):
    """Vehicle model matching the main application schema."""
    __tablename__ = "vehicle"

    id: Mapped[int] = mapped_column(primary_key=True)
    id_crabi: Mapped[str] = mapped_column(String, index=True, unique=True)
    description: Mapped[str] = mapped_column(String, index=False)


# ============================================================================
# Services
# ============================================================================

class EmbeddingService:
    """
    Embedding service using HuggingFace models.
    Replicates the embedding calculation from the main application.
    """
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        logger.info(f"Initializing HuggingFace Embeddings with model: {model_name}")
        self.model = HuggingFaceEmbeddings(model_name=model_name)
    
    def calculate_embedding(self, text: str) -> list[float]:
        """Calculate embedding for a single text string."""
        if not text:
            return []
        return self.model.embed_query(text)
    
    def calculate_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        """Calculate embeddings for multiple texts in batch."""
        if not texts:
            return []
        return self.model.embed_documents(texts)


class VectorRepository:
    """
    Repository for Upstash Vector database operations.
    Uses batch upsert for efficient bulk operations.
    """
    
    def __init__(self, base_url: str, token: str, timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout
    
    @property
    def headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
    
    def upsert_batch(
        self,
        vectors: list[dict[str, Any]],
        namespace: str | None = None
    ) -> dict[str, Any]:
        """
        Upsert multiple vectors in a single request.
        
        Args:
            vectors: List of vector objects with id, vector, and optional metadata
            namespace: Optional namespace for the vectors
        
        Returns:
            Response from the Upstash API
        """
        url = f"{self.base_url}/upsert"
        if namespace:
            url = f"{self.base_url}/upsert/{namespace}"
        
        response = requests.post(
            url,
            json=vectors,
            headers=self.headers,
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()


# ============================================================================
# Data Processing
# ============================================================================

@dataclass
class VehicleRecord:
    """Represents a vehicle record from CSV."""
    description: str
    id_crabi: str


def read_csv(file_path: str) -> list[VehicleRecord]:
    """
    Read vehicle records from a CSV file.
    
    Expected CSV format:
    versionc,id_crabi
    "FIAT MOBI 2024 TREKKING, L4, 1.0L, 69 CP, 5 PUERTAS, AUT",FM-100
    """
    records = []
    
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            # Handle both possible column names
            description = row.get("versionc") or row.get("description") or ""
            id_crabi = row.get("id_crabi") or ""
            
            if description and id_crabi:
                records.append(VehicleRecord(
                    description=description.strip(),
                    id_crabi=id_crabi.strip()
                ))
            else:
                logger.warning(f"Skipping invalid row: {row}")
    
    logger.info(f"Read {len(records)} records from CSV")
    return records


def populate_postgres(
    records: list[VehicleRecord],
    session: Session,
    batch_size: int = 100
) -> None:
    """
    Populate PostgreSQL database with vehicle records using batch inserts.
    
    Args:
        records: List of vehicle records to insert
        session: SQLAlchemy session
        batch_size: Number of records per batch
    """
    logger.info(f"Populating PostgreSQL with {len(records)} records...")
    
    total_inserted = 0
    
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        
        vehicles = [
            Vehicle(
                id_crabi=record.id_crabi,
                description=record.description
            )
            for record in batch
        ]
        
        session.add_all(vehicles)
        session.commit()
        
        total_inserted += len(vehicles)
        logger.info(f"Inserted batch {i // batch_size + 1}: {total_inserted}/{len(records)} records")
    
    logger.info(f"Successfully populated PostgreSQL with {total_inserted} records")


def populate_vector_db(
    records: list[VehicleRecord],
    embedding_service: EmbeddingService,
    vector_repo: VectorRepository,
    batch_size: int = 50,
    namespace: str | None = None
) -> None:
    """
    Populate Upstash Vector database with vehicle embeddings.
    
    Args:
        records: List of vehicle records
        embedding_service: Service for calculating embeddings
        vector_repo: Repository for vector operations
        batch_size: Number of records per batch (smaller for embeddings due to computation)
        namespace: Optional namespace for vectors
    """
    logger.info(f"Populating Vector DB with {len(records)} records...")
    
    total_upserted = 0
    
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        
        # Extract texts for batch embedding calculation
        texts = [record.description for record in batch]
        
        # Calculate embeddings in batch
        logger.info(f"Calculating embeddings for batch {i // batch_size + 1}...")
        embeddings = embedding_service.calculate_embeddings_batch(texts)
        
        # Prepare vectors for upsert
        vectors = [
            {
                "id": record.id_crabi,
                "vector": embedding,
                "metadata": {
                    "description": record.description,
                    "id_crabi": record.id_crabi
                }
            }
            for record, embedding in zip(batch, embeddings)
        ]
        
        # Upsert batch to vector database
        logger.info(f"Upserting batch {i // batch_size + 1} to vector database...")
        result = vector_repo.upsert_batch(vectors, namespace=namespace)
        logger.info(f"Upsert result: {result}")
        
        total_upserted += len(vectors)
        logger.info(f"Upserted {total_upserted}/{len(records)} vectors")
    
    logger.info(f"Successfully populated Vector DB with {total_upserted} vectors")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Populate PostgreSQL and Vector databases from CSV"
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to the CSV file containing vehicle data"
    )
    parser.add_argument(
        "--postgres-batch-size",
        type=int,
        default=100,
        help="Batch size for PostgreSQL inserts (default: 100)"
    )
    parser.add_argument(
        "--vector-batch-size",
        type=int,
        default=50,
        help="Batch size for vector embeddings (default: 50)"
    )
    parser.add_argument(
        "--namespace",
        type=str,
        default=None,
        help="Optional namespace for vector database"
    )
    parser.add_argument(
        "--skip-postgres",
        action="store_true",
        help="Skip PostgreSQL population"
    )
    parser.add_argument(
        "--skip-vectors",
        action="store_true",
        help="Skip vector database population"
    )
    
    args = parser.parse_args()
    
    # Load and validate configuration
    logger.info("Loading configuration from environment variables...")
    config = Config.from_env()
    
    try:
        config.validate()
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)
    
    # Read CSV data
    logger.info(f"Reading CSV file: {args.csv}")
    records = read_csv(args.csv)
    
    if not records:
        logger.error("No valid records found in CSV file")
        sys.exit(1)
    
    # Populate PostgreSQL
    if not args.skip_postgres:
        logger.info("Connecting to PostgreSQL...")
        engine = create_engine(config.db_url)
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        
        with SessionLocal() as session:
            try:
                populate_postgres(records, session, batch_size=args.postgres_batch_size)
            except Exception as e:
                logger.error(f"Failed to populate PostgreSQL: {e}")
                session.rollback()
                raise
    else:
        logger.info("Skipping PostgreSQL population")
    
    # Populate Vector Database
    if not args.skip_vectors:
        logger.info("Initializing embedding service...")
        embedding_service = EmbeddingService(config.embedding_model)
        
        logger.info("Initializing vector repository...")
        vector_repo = VectorRepository(
            base_url=config.upstash_vector_rest_url,
            token=config.upstash_vector_rest_token
        )
        
        try:
            populate_vector_db(
                records,
                embedding_service,
                vector_repo,
                batch_size=args.vector_batch_size,
                namespace=args.namespace
            )
        except Exception as e:
            logger.error(f"Failed to populate vector database: {e}")
            raise
    else:
        logger.info("Skipping vector database population")
    
    logger.info("Database population completed successfully!")


if __name__ == "__main__":
    main()

