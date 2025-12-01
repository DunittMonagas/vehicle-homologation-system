import logging
from langchain_huggingface import HuggingFaceEmbeddings
from app.core.config import config

logger = logging.getLogger(__name__)

class EmbeddingService:
    def __init__(self):
        self.model_name = config.embedding_model
        self.model = self._initialize_model()

    def _initialize_model(self):
        try:
            logger.info(f"Initializing Local Embeddings with model: {self.model_name}")
            return HuggingFaceEmbeddings(
                model_name=self.model_name
            )
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            raise

    def calculate_embedding(self, text: str) -> list[float]:
        """
        Calculates the embedding for a given text string.
        """
        if not text:
            return []
        
        try:
            return self.model.embed_query(text)
        except Exception as e:
            logger.error(f"Error calculating embedding: {e}")
            raise
