from typing import List
from llm.RAG.Embeddings.BaseEmbedding import BaseEmbeddings


class VectorStore:
    def __init__(self, document: List[str] = [""]) -> None:
        self.document = document

    def get_vector(self, EmbeddingModel: BaseEmbeddings) -> List[List[float]]:
        # 获得文档的向量表示
        pass

    def persist(self, path: str = "storage"):
        # 数据库持久化保存
        pass

    def load_vector(self, path: str = "storage"):
        # 从本地加载数据库
        pass

    def query(
        self, query: str, EmbeddingModel: BaseEmbeddings, k: int = 1
    ) -> List[str]:
        # 根据问题检索相关文档片段
        pass
