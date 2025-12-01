from typing import Any, TypedDict
import json
import logging

import requests

from app.core.config import config


class VectorQueryResult(TypedDict):
    id: str
    score: float
    metadata: dict[str, Any] | None
    vector: list[float] | None


class VectorQueryResponse(TypedDict):
    result: list[VectorQueryResult]


class VectorRepository:
    def __init__(
        self,
        timeout: float = 30.0
    ):
        self.base_url = config.upstash_vector_rest_url.rstrip('/')
        self.timeout = timeout

    @property
    def headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {config.upstash_vector_rest_token}",
            "Content-Type": "application/json"
        }

    def query(
        self,
        vector: list[float],
        top_k: int = 10,
        include_metadata: bool = True,
        include_vectors: bool = False,
        filter: str | None = None,
        namespace: str | None = None
    ) -> VectorQueryResponse:
        url = f"{self.base_url}/query"
        
        payload: dict[str, Any] = {
            "vector": vector,
            "topK": top_k,
            "includeMetadata": include_metadata,
            "includeVectors": include_vectors
        }

        if filter:
            payload["filter"] = filter
        if namespace:
            payload["namespace"] = namespace

        # logging.info(payload)

        response = requests.post(
            url,
            json=payload,
            headers=self.headers,
            timeout=self.timeout
        )

        response.raise_for_status()
        result = response.json()
        # logging.info(f"Vector query response:\n{json.dumps(result, indent=2)}")
        return result

