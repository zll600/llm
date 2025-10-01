from openai import OpenAI
from llm.RAG.LLM.BaseModel import BaseModel
import os
from typing import List

RAG_PROMPT_TEMPLATE = """
使用以上下文来回答用户的问题。如果你不知道答案，就说你不知道。总是使用中文回答。
问题: {question}
可参考的上下文：
···
{context}
···
如果给定的上下文无法让你做出回答，请回答数据库中没有这个内容，你不知道。
有用的回答:
"""


class OpenAIChat(BaseModel):
    def __init__(self, model: str = "Qwen/Qwen2.5-32B-Instruct") -> None:
        self.model = model

    def chat(self, prompt: str, history: List[dict], content: str) -> str:
        client = OpenAI()
        client.api_key = os.getenv("OPENAI_API_KEY")
        client.base_url = os.getenv("OPENAI_BASE_URL")
        history.append(
            {
                "role": "user",
                "content": RAG_PROMPT_TEMPLATE.format(question=prompt, context=content),
            }
        )
        response = client.chat.completions.create(
            model=self.model, messages=history, max_tokens=2048, temperature=0.1
        )
        return response.choices[0].message.content
