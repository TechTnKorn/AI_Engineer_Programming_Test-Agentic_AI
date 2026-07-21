"""Two-agent RAG pipeline: Data Retriever -> Report Generator.

The Data Retriever agent is a small ReAct agent bound to a single tool
(`search_knowledge_base`) that performs the actual retrieval. The tool result
is fed back into the retriever LLM (standard ReAct loop), but the *raw* tool
output — not the retriever's paraphrase — is what gets handed off to the
Report Generator agent, which is a plain LLM call (no tools) that synthesizes
the final answer.
"""

from typing import Dict

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.runtime import Runtime
from langgraph.prebuilt import ToolNode

from react_agent.context import Context
from react_agent.state import InputState, State
from react_agent.tools import search_knowledge_base, TOOLS
from utils import select_llm_provider

tool_node = ToolNode(TOOLS)

retriever_model = select_llm_provider(temperature=0.0, reasoning_effort=None, streaming=False).bind_tools([search_knowledge_base])
generator_model = select_llm_provider(temperature=0.3, reasoning_effort=None, streaming=True)

def latest_query(state: State) -> str:
    """Pull the end-user's question out of the message history."""
    for message in reversed(state.get("messages") or []):
        if isinstance(message, HumanMessage):
            return str(message.content)
    return ""


async def retrieve(state: State, runtime: Runtime[Context]) -> Dict[str, list]:
    """Run the Data Retriever agent and append its (tool-calling) reply to state."""
    messages = state.get("messages") or []
    response = await retriever_model.ainvoke(
        [SystemMessage(content=runtime.context.retriever_system_prompt)] + messages
    )

    # hide success msg
    if not response.tool_calls:
        return

    return {"messages": [response]}



async def generate(state: State, runtime: Runtime[Context]) -> Dict[str, str]:
    """Run the Report Generator agent over the retrieved snippets."""

    snippets = "\n\n".join(
        str(message.content)
        for message in state.get("messages") or []
        if isinstance(message, ToolMessage) and message.content
    )

    if not snippets:
        snippets = "No relevant information was found in the knowledge base."

    system_message = runtime.context.generator_system_prompt.format(
        query=latest_query(state), snippets=snippets
    )

    response = await generator_model.ainvoke([SystemMessage(content=system_message)])
    return {"messages": [response], "snippets": snippets}

def tool_router(state: State):
    """Route to tools if LLM made a tool call, otherwise go straight to save."""
    messages = state.get("messages") or []
    if messages and getattr(messages[-1], "tool_calls", None):
        return "tools"
    return "generate"

builder = StateGraph(State, input_schema=InputState, context_schema=Context)

builder.add_node("retrieve", retrieve)
builder.add_node("tools", tool_node)
builder.add_node("generate", generate)

builder.add_edge("__start__", "retrieve")
builder.add_conditional_edges("retrieve", tool_router, {"tools": "tools", "generate": "generate"})
builder.add_edge("tools", "retrieve")
builder.add_edge("generate", "__end__")

graph = builder.compile(name="RAG Agent (Retriever -> Report Generator)")
