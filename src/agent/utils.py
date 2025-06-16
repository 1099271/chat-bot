from langchain_community.tools import BraveSearch
import os
from dotenv import load_dotenv
from langgraph.graph import END
from schemas import State
from langgraph.types import Command, interrupt
from langchain_core.tools import tool

load_dotenv()


def get_brave_search_tool():
    return BraveSearch.from_api_key(os.getenv("BRAVE_SEARCH_API_KEY"), max_results=10)


def route_tools(state: State):
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages in state to tool_edge: {state}")

    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END

@tool
def human_assistance(query: str) -> str:
    """Ask for human assistance."""
    human_response = interrupt({"query": query})
    return human_response