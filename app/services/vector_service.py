import json
import logging
from app.repositories.vector_repository import VectorRepository
from app.services.embedding_service import EmbeddingService

class VectorService:
    def __init__(self, vector_repository: VectorRepository, embedding_service: EmbeddingService):
        self.vector_repository = vector_repository
        self.embedding_service = embedding_service

    def calculate_embedding(self, description: str):
        return self.embedding_service.calculate_embedding(description)

    def query(self, vector: list[float], top_k: int = 10):
        return self.vector_repository.query(vector, top_k)

    def query_by_description(self, description: str, top_k: int = 10):
        vector = self.calculate_embedding(description)
        # logging.info(f"query_by_description vector: {vector[:5]}")
        result = self.query(vector, top_k)
        # logging.info(f"query_by_description result:\n{json.dumps(result, indent=2)}")
        return result
