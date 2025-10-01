from Embeddings.BaseEmbedding import BaseEmbeddings
import os
from openai import OpenAI
from typing import List


class OpenAIEmbedding(BaseEmbeddings):
    def __init__(self, path: str = "", is_api: bool = True) -> None:
        super().__init__(path, is_api)
        if self.is_api:
            self.client = OpenAI()
            self.client.api_key = os.getenv("OPENAI_API_KEY")
            self.client.base_url = os.getenv("OPENAI_BASE_URL")

    def get_embedding(
        self, text: str, model: str = "text-embedding-3-small"
    ) -> List[float]:
        if self.is_api:
            text = text.replace("\n", " ")
            try:
                response = self.client.embeddings.create(input=[text], model=model)
                if hasattr(response, "data") and response.data:
                    return response.data[0].embedding
                else:
                    raise ValueError(
                        f"Invalid API response format. Got: {type(response)}"
                    )
            except Exception as e:
                print("\n[ERROR] Failed to get embedding:")
                print(f"  Error type: {type(e).__name__}")
                print(f"  Error message: {str(e)}")
                raise
        else:
            raise NotImplementedError
