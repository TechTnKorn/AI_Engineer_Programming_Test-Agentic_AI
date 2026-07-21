"""Two-agent RAG pipeline: Data Retriever -> Report Generator.

The Data Retriever agent is a small ReAct agent bound to a single tool
(`search_knowledge_base`) that performs the actual retrieval; its raw tool
output (not its own paraphrase) is handed off to the Report Generator agent,
which is a plain LLM call with no tools that synthesizes a final answer.
"""

from typing import Dict

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph
from langgraph.runtime import Runtime
from langgraph.prebuilt import ToolNode

from react_agent.context import Context
from react_agent.state import InputState, State
from react_agent.tools import search_knowledge_base
from react_agent.tools import TOOLS
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
    response = await retriever_model.ainvoke(
        [SystemMessage(content=runtime.context.retriever_system_prompt)]
        + [HumanMessage(content=latest_query(state))]
    )

    return {"messages": [response]}


async def generate(state: State, runtime: Runtime[Context]) -> Dict[str, str]:
    """Run the Report Generator agent over the retrieved snippets."""

    # The raw tool output lives in the ToolMessage(s) the `tools` node appended.
    snippets = "\n\n".join(
        str(message.content)
        for message in state.get("messages") or []
        if isinstance(message, ToolMessage)
    )

    system_message = runtime.context.generator_system_prompt.format(
        query=latest_query(state), snippets=snippets
    )

    response = await generator_model.ainvoke([SystemMessage(content=system_message)])

    return {"messages": str(response.content)}

def tool_router(state: State):
    """Route to tools if LLM made a tool call, otherwise go straight to save."""
    messages = state.get("messages") or []
    if not messages:
        return "generate"
    # An AIMessage that calls a tool usually has *empty* .content, so the tool
    # calls themselves are the only reliable signal here.
    return "tools" if getattr(messages[-1], "tool_calls", None) else "generate"

builder = StateGraph(State, input_schema=InputState, context_schema=Context)

builder.add_node("retrieve", retrieve)
builder.add_node("generate", generate)
builder.add_node("tools", tool_node)

builder.add_edge("__start__", "retrieve")
builder.add_conditional_edges("retrieve", tool_router, {"tools": "tools", "generate": "generate"})
builder.add_edge("tools", "generate")

builder.add_edge("generate", "__end__")

graph = builder.compile(name="RAG Agent (Retriever -> Report Generator)")
