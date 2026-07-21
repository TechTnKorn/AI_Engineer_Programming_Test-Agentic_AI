import numpy as np
from langchain_core.tools import tool
from langgraph.runtime import get_runtime
from sklearn.metrics.pairwise import cosine_similarity

from react_agent.context import Context
from utils import select_embedding_provider

characters_per_chunk = 1000
overlap_characters = 100

# Read file
with open("knowledge_base.txt", "r") as f:
    raw_text = f.read()
raw_text = raw_text.strip().replace("\n", " ")

# Chunking
knowledge_base = []
for i in range(0, len(raw_text), characters_per_chunk):
    start = max(0, i - overlap_characters)
    knowledge_base.append(raw_text[start : i + characters_per_chunk])

chunks = ""
for i, chunk in enumerate(knowledge_base):
    chunks += f"Chunk {i}: {chunk}\n"

# Write to file
with open("knowledge_base_chunk.txt", "w") as f:
    f.write(chunks)

# Embedding
embedder = select_embedding_provider()
chunk_embeddings = np.array(embedder.embed_documents(knowledge_base))

@tool(description="Search knowledge base for paragraphs relevant to a query.")
def search_knowledge_base(query: str) -> str:
    """Search knowledge_base.txt for the paragraphs most relevant to a query.

    Args:
        query: The user's question or topic to search for.
    """
    runtime = get_runtime(Context)
    top_k = int(runtime.context.top_k)

    if not knowledge_base:
        return "No relevant information was found in the knowledge base."

    scores = cosine_similarity(chunk_embeddings, [embedder.embed_query(query)]).ravel()
    ranked = np.argsort(scores)[::-1][:top_k]

    return "\n\n".join(knowledge_base[index] for index in ranked)


TOOLS = [search_knowledge_base]
