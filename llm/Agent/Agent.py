from openai import OpenAI
from typing import List, Dict, Any
from utils import function_to_json


SYSTEM_PROMPT = """
You are an AI assistant named "Song". Your output should be consistent with the user's language.
When the user's question requires calling a tool, you can call the appropriate tool function from the provided list of tools.
"""


class Agent:
    def __init__(
        self,
        client: OpenAI,
        model: str,
        tools: List = [],
        verbose: bool = True,
    ):
        self.client = client
        self.tools = tools
        self.model = model
        self.messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]
        self.verbose = verbose

    def get_tool_schema(self) -> List[Dict[str, Any]]:
        # 获取所有工具的 JSON 模式
        return [function_to_json(tool) for tool in self.tools]

    def handle_tool_call(self, tool_call):
        function_name = tool_call.function.name
        function_args = tool_call.function.arguments
        function_id = tool_call.id

        function_call_content = eval(f"utils.{function_name}(**{function_args})")

        return {
            "role": "tool",
            "content": function_call_content,
            "tool_call_id": function_id,
        }

    def get_completion(self, prompt) -> str:
        self.messages.append({"role": "user", "content": prompt})

        # 获取模型的完成响应
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            tools=self.get_tool_schema(),
            stream=False,
        )
        if response.choices[0].message.tool_calls:
            self.messages.append(
                {"role": "assistant", "content": response.choices[0].message.content}
            )
            tool_list = []
            for tool_call in response.choices[0].message.tool_calls:
                # 处理工具调用并将结果添加到消息列表中
                self.messages.append(self.handle_tool_call(tool_call))
                tool_list.append(
                    [tool_call.function.name, tool_call.function.arguments]
                )
            if self.verbose:
                print("calling tools: ", response.choices[0].message.content, tool_list)
            # 再次获取模型的完成响应，这次包含工具调用的结果
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=self.get_tool_schema(),
                stream=False,
            )

        # 将模型的完成响应添加到消息列表中
        self.messages.append(
            {"role": "assistant", "content": response.choices[0].message.content}
        )
        return response.choices[0].message.content
