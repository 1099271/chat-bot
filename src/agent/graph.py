import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from utils import get_brave_search_tool, route_tools, human_assistance

from schemas import BasicToolNode, State
from loguru import logger

# from langgraph.types import Command, interrupt
# from langchain_core.tools import tool

load_dotenv()
if os.getenv("DEEPSEEK_KEY") is None:
    raise ValueError("DEEPSEEK_KEY is not set")


logger.info("Starting LangGraph")
logger.info("llm model: deepseek-chat")
graph_builder = StateGraph(State)
api_key = os.getenv("DEEPSEEK_KEY")
llm = ChatOpenAI(
    model="deepseek-chat",
    base_url="https://api.deepseek.com/v1",
    api_key=api_key,
)
tools = [get_brave_search_tool(), human_assistance]
llm_with_tools = llm.bind_tools(tools)
# tool_node = BasicToolNode(tools=tools)


def chatbot(state: State):
    message = llm_with_tools.invoke(state["messages"])
    # 记录工具调用数量（用于调试）
    if hasattr(message, "tool_calls") and message.tool_calls:
        logger.info(f"Tool calls: {len(message.tool_calls)}")
        for i, tool_call in enumerate(message.tool_calls):
            logger.info(f"  {i+1}. {tool_call['name']}: {tool_call['args']}")
    return {"messages": [message]}


graph_builder.add_node("chatbot", chatbot)
tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)
# graph_builder.add_conditional_edges(
#     "chatbot", route_tools, {"tools": "tools", END: END}
# )
graph_builder.add_conditional_edges("chatbot", tools_condition)

graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)
