from langchain_core.tools import tool
from langgraph.runtime import get_runtime
from langchain_community.retrievers import BM25Retriever

from react_agent.context import Context

characters_per_chunk = 500
overlap_characters = 50

with open("knowledge_base.txt", "r") as f:
    raw_text = f.read()
raw_text = raw_text.strip().replace("\n", " ")

# Chunking
knowledge_base = []
for i in range(0, len(raw_text), characters_per_chunk):
    if i + characters_per_chunk < len(raw_text):
        if i > 0:
            knowledge_base.append(
                raw_text[i - overlap_characters : i + characters_per_chunk]
            )    
        else:
            knowledge_base.append(raw_text[i : i + characters_per_chunk])
    elif i + characters_per_chunk > len(raw_text):
        knowledge_base.append(raw_text[i : len(raw_text)])

chunks = ""
for i, chunk in enumerate(knowledge_base):
    # print(f"Chunk {i}: {chunk}")
    chunks += f"Chunk {i}: {chunk}\n"    

# Write to file
with open("knowledge_base_chunk.txt", "w") as f:
    f.write(chunks)

bm25_retriever = BM25Retriever.from_texts(knowledge_base)
bm25_retriever.k = len(knowledge_base)


@tool(description="Search knowledge base for paragraphs relevant to a query.")
def search_knowledge_base(query: str) -> str:
    """Search knowledge_base.txt for the paragraphs most relevant to a query.

    Args:
        query: The user's question or topic to search for.
    """
    runtime = get_runtime(Context)
    top_k = runtime.context.top_k

    candidates = bm25_retriever.invoke(query)[:top_k]
    if not candidates:
        return "No relevant information was found in the knowledge base."
    print(candidates)
    return "\n\n".join(doc.page_content for doc in candidates)


TOOLS = [search_knowledge_base]
