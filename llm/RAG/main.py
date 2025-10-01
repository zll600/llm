from VectorBase import VectorStore
from utils import ReadFiles
from LLM.OpenAIChat import OpenAIChat
from Embeddings.OpenAiEmbedding import OpenAIEmbedding
import os

embedding = OpenAIEmbedding()
# 没有保存数据库
# 获得data目录下的所有文件内容并分割
vector = None
if os.path.exists("./storage/document.json") and os.path.exists(
    "./storage/vectors.json"
):
    vector = VectorStore()
    vector.load_vector("./storage")
else:
    docs = ReadFiles("./data").get_content(max_token_len=600, cover_content=150)
    vector = VectorStore(docs)
    vector.get_vector(EmbeddingModel=embedding)
    vector.persist(path="storage")

question = "How RAG works?"

content = vector.query(question, EmbeddingModel=embedding, k=1)[0]
chat = OpenAIChat()
print(chat.chat(question, [], content))
