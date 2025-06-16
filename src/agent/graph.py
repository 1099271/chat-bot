import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from utils import get_brave_search_tool, route_tools, human_assistance

from schemas import BasicToolNode, State
from loguru import logger

# from langgraph.types import Command, interrupt
# from langchain_core.tools import tool

load_dotenv()
if os.getenv("OPENROUTER_KEY") is None:
    raise ValueError("OPENROUTER_KEY is not set")


logger.info("Starting LangGraph")
logger.info("llm model: o4-mini-2025-04-16")
graph_builder = StateGraph(State)
api_key = os.getenv("OPENROUTER_KEY")
llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)
tools = [get_brave_search_tool(), human_assistance]
llm_with_tools = llm.bind_tools(tools)
# tool_node = BasicToolNode(tools=tools)



def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


graph_builder.add_node("chatbot", chatbot)
tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)
# graph_builder.add_conditional_edges(
#     "chatbot", route_tools, {"tools": "tools", END: END}
# )
graph_builder.add_conditional_edges("chatbot", tools_condition)

graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

graph = graph_builder.compile()
