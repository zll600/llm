from openai import OpenAI
from LLM.BaseModel import BaseModel
import os
from typing import List

RAG_PROMPT_TEMPLATE = """
Use the following context to answer the user's question. If you don't know the answer, just say that you don't know.
question: {question}
available context：
···
{context}
···
Answer you don't know if the answer is not in the context.
Available useful answers:
"""


class OpenAIChat(BaseModel):
    def __init__(self, model: str = "gpt-4.1") -> None:
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
