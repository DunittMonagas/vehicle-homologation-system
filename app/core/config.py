from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from app.core.constants import EmbeddingModel, GeminiModel

load_dotenv()


class Config(BaseSettings):
    app_name: str = "VehicleHomologationSystem"
    debug: bool = False
    db_user: str = ""
    db_password: str = ""
    db_name: str = ""
    db_host: str = ""
    db_port: int = 5432
    
    # Upstash Vector configuration
    upstash_vector_rest_url: str = ""
    upstash_vector_rest_token: str = ""

    # Embedding configuration
    embedding_model: str = EmbeddingModel.LOCAL_ALL_MINILM_L6_V2.value

    # Gemini LLM configuration
    gemini_api_key: str = ""
    gemini_model: str = GeminiModel.GEMINI_2_5_PRO.value

    # Vehicle matching configuration
    vector_similarity_threshold: float = 0.85
    vector_similarity_threshold_best_effort: float = 0.70
    vector_top_k: int = 10

    @property
    def db_url(self):
        return f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"


config = Config()