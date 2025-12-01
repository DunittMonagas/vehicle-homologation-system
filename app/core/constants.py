from enum import Enum


class EmbeddingModel(str, Enum):
    LOCAL_ALL_MINILM_L6_V2 = "all-MiniLM-L6-v2"


class GeminiModel(str, Enum):
    GEMINI_2_5_PRO = "gemini-2.5-pro"
    GEMINI_2_5_FLASH_LITE = "gemini-2.5-flash-lite"
    GEMINI_2_5_FLASH = "gemini-2.5-flash"
    GEMINI_3_PRO_PREVIEW = "gemini-3-pro-preview"


VALID_GEMINI_MODELS = [model.value for model in GeminiModel]
