from enum import Enum


class EmbeddingProvider(str, Enum):
    OPENAI = "openai"
    LOCAL = "local"


class EmbeddingModel(str, Enum):
    OPENAI_TEXT_EMBEDDING_3_SMALL = "text-embedding-3-small"
    LOCAL_ALL_MINILM_L6_V2 = "all-MiniLM-L6-v2"


class GeminiModel(str, Enum):
    GEMINI_2_5_PRO = "gemini-2.5-pro"
    GEMINI_2_5_FLASH_LITE = "gemini-2.5-flash-lite"
    GEMINI_2_5_FLASH = "gemini-2.5-flash"
    GEMINI_3_PRO_PREVIEW = "gemini-3-pro-preview"


MODEL_TO_PROVIDER = {
    EmbeddingModel.OPENAI_TEXT_EMBEDDING_3_SMALL.value: EmbeddingProvider.OPENAI,
    EmbeddingModel.LOCAL_ALL_MINILM_L6_V2.value: EmbeddingProvider.LOCAL,
}


VALID_GEMINI_MODELS = [model.value for model in GeminiModel]
