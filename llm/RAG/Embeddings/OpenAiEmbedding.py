from llm.RAG.Embeddings.BaseEmbedding import BaseEmbeddings


class OpenAIEmbedding(BaseEmbeddings):
    """
    class for OpenAI embeddings
    """

    def __init__(self, path: str = "", is_api: bool = True) -> None:
        super().__init__(path, is_api)
        if self.is_api:
            self.client = OpenAI()
            # 从环境变量中获取 硅基流动 密钥
            self.client.api_key = os.getenv("OPENAI_API_KEY")
            # 从环境变量中获取 硅基流动 的基础URL
            self.client.base_url = os.getenv("OPENAI_BASE_URL")

    def get_embedding(self, text: str, model: str = "BAAI/bge-m3") -> List[float]:
        """
        此处默认使用轨迹流动的免费嵌入模型 BAAI/bge-m3
        """
        if self.is_api:
            text = text.replace("\n", " ")
            return (
                self.client.embeddings.create(input=[text], model=model)
                .data[0]
                .embedding
            )
        else:
            raise NotImplementedError
