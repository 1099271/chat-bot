import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from utils import get_brave_search_tool, route_tools
from schemas import BasicToolNode, State
from loguru import logger


load_dotenv()
if os.getenv("OPENROUTER_KEY") is None:
    raise ValueError("OPENROUTER_KEY is not set")


logger.info("Starting LangGraph")
logger.info("llm model: o4-mini-2025-04-16")
graph_builder = StateGraph(State)
llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_KEY"),
)
tool = get_brave_search_tool()
tools = [tool]
llm_with_tools = llm.bind_tools(tools)
# tool_node = BasicToolNode(tools=tools)
tool_node = ToolNode(tools=tools)


def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}


graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)
# graph_builder.add_conditional_edges(
#     "chatbot", route_tools, {"tools": "tools", END: END}
# )
graph_builder.add_conditional_edges("chatbot", tools_condition)

graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

graph = graph_builder.compile()
