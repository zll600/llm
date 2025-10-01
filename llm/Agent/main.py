from openai import OpenAI
import os
from Agent import Agent
from tools import get_current_datetime, search_wikipedia, get_current_temperature

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"), base_url=os.environ.get("OPENAI_BASE_URL")
)

agent = Agent(
    client=client,
    model="gpt-4.1",
    tools=[get_current_datetime, search_wikipedia, get_current_temperature],
)

while True:
    # 使用彩色输出区分用户输入和AI回答
    prompt = input("\033[94mUser: \033[0m")  # 蓝色显示用户输入提示
    if prompt == "exit":
        break
    response = agent.get_completion(prompt)
    print("\033[92mAssistant: \033[0m", response)  # 绿色显示AI助手回答
