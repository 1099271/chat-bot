import json
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import ToolMessage


class State(TypedDict):
    messages: Annotated[list, add_messages]


class BasicToolNode:
    """
    A node that runs the tools requested in the last AIMessage.
    """

    def __init__(self, tools: list) -> None:
        self.tool_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        if messages := inputs.get("messages"):
            message = messages[-1]
        else:
            raise ValueError("No messages in inputs")

        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tool_by_name[tool_call["name"]].invoke(tool_call["args"])
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
