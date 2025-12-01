import logging
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from app.core.config import config
from app.core.constants import EmbeddingProvider, MODEL_TO_PROVIDER

logger = logging.getLogger(__name__)

class EmbeddingService:
    def __init__(self):
        self.model_name = config.embedding_model
        self.provider = MODEL_TO_PROVIDER.get(self.model_name)
        
        if not self.provider:
             raise ValueError(f"Unsupported embedding model: {self.model_name}")

        self.model = self._initialize_model()

    def _initialize_model(self):
        try:
            if self.provider == EmbeddingProvider.OPENAI:
                logger.info(f"Initializing OpenAI Embeddings with model: {self.model_name}")
                return OpenAIEmbeddings(
                    api_key=config.llm_api_key,
                    model=self.model_name
                )
            elif self.provider == EmbeddingProvider.LOCAL:
                logger.info(f"Initializing Local Embeddings with model: {self.model_name}")
                return HuggingFaceEmbeddings(
                    model_name=self.model_name
                )
            else:
                # This should technically be unreachable due to the check in __init__
                raise ValueError(f"Unsupported embedding provider: {self.provider}")
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
            result = self.model.embed_query(text)
            # logging.info(f"calculate_embedding len: {len(result)}, result: {result[:5]}")
            return result
        except Exception as e:
            logger.error(f"Error calculating embedding: {e}")
            raise
