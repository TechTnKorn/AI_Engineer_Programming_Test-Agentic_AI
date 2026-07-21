# Multi-Agent RAG: Data Retriever → Report Generator

A minimal two-agent RAG pipeline built with [LangGraph](https://github.com/langchain-ai/langgraph), scaffolded from the official [LangGraph ReAct Agent template](https://github.com/langchain-ai/react-agent) (hence the `src/react_agent/` package name).

- **Data Retriever** — a tool-calling agent whose only job is to search a local knowledge base (`knowledge_base.txt`) and return raw, relevant text chunks. It never answers the question itself.
- **Report Generator** — a plain LLM call (no tools) that receives those chunks and synthesizes one cohesive, non-redundant, well-formatted answer.

The two are orchestrated as a **sequential handoff**: the Retriever's raw *tool output* (not its own paraphrase) is passed into the Generator.

Retrieval uses **vector search** — the knowledge base is chunked, embedded once at import time, and each query is matched by cosine similarity, returning the `top_k` best chunks.

## Graph

```text
__start__ → retrieve ─(tool_calls?)─→ tools ─→ generate → __end__
                    └──────(no)───────────────↗
```

| Node | File | What it does |
| --- | --- | --- |
| `retrieve` | [graph.py](src/react_agent/graph.py) | Data Retriever agent; decides to call `search_knowledge_base` |
| `tools` | [tools.py](src/react_agent/tools.py) | Executes the vector search over the chunked knowledge base |
| `generate` | [graph.py](src/react_agent/graph.py) | Report Generator agent; writes the final answer from the retrieved chunks |

## Project layout

```
├── knowledge_base.txt          # the source knowledge base (edit this)
├── knowledge_base_chunk.txt    # generated: chunk dump, for inspection/debugging
├── utils.py                    # LLM + embedding provider selection (Azure / Ollama)
├── langgraph.json              # exposes the `rag_agent` graph to LangGraph Studio
└── src/react_agent/
    ├── graph.py                # nodes, routing, graph assembly
    ├── tools.py                # chunking, embedding, search_knowledge_base tool
    ├── prompts.py              # system prompts for both agents
    ├── context.py              # runtime-configurable settings (prompts, top_k)
    └── state.py                # graph state schema
```

## Setup

### 1. Create the environment

```bash
conda create -n {env name} python=3.12
conda activate {env name}

python -m pip install --upgrade pip
pip install uv
```

### 2. Install dependencies

```bash
uv pip install -U "langgraph-cli[inmem]"
uv pip install -e .
```

`uv pip install -e .` installs the project itself plus everything declared in [pyproject.toml](pyproject.toml) — `langgraph`, `langchain`, `langchain-openai`, `langchain-ollama`, `scikit-learn`, `sentence-transformers`, and so on. Installing it editable (`-e`) is what makes `import react_agent` and `import utils` resolve when the graph is loaded.

### 3. Configure environment variables

```bash
cp .env.example .env
```

Then fill in `.env`:

| Variable | Notes |
| --- | --- |
| `LLM_PROVIDER` | `azure` or `ollama` |
| `EMBEDDING_PROVIDER` | `azure` or `ollama` |
| `AZURE_ENDPOINT` | e.g. `https://<resource>.openai.azure.com/` |
| `AZURE_DEPLOYMENT` | chat model deployment name (e.g. `gpt-5-mini`) |
| `AZURE_EMBEDDING_MODEL` | embedding deployment name (e.g. `text-embedding-3-small`) |
| `AZURE_OPENAI_KEY` | Azure OpenAI API key |
| `AZURE_OPENAI_VERSION` | e.g. `2024-10-21` |
| `OLLAMA_BASE_URL` | e.g. `http://localhost:11434` |
| `OLLAMA_MODEL` | chat model, e.g. `qwen3:8b` |
| `OLLAMA_EMBEDDING_MODEL` | embedding model, e.g. `nomic-embed-text` |

Only the block matching your chosen provider needs to be filled in. Provider selection happens in [utils.py](utils.py) — `select_llm_provider()` and `select_embedding_provider()`.

### 4. Run

```bash
langgraph dev
```

This starts the in-memory LangGraph server and opens LangGraph Studio, where you can send a query to the `rag_agent` graph and watch both agents execute. Local edits hot-reload.

## Usage

Ask a question about whatever is in `knowledge_base.txt`:

```
What is the policy on international travel?
```

The Retriever pulls the most similar chunks; the Generator turns them into the final answer.

To use your own data, replace `knowledge_base.txt` and restart — chunking and embedding run at import time, so a restart is required for changes to take effect. `knowledge_base_chunk.txt` is regenerated on each start and is only there so you can eyeball how the text was split.

## Tuning

| Setting | Where | Default |
| --- | --- | --- |
| `top_k` — chunks returned per query | [context.py](src/react_agent/context.py) (or via Studio context / `TOP_K` env var) | `3` |
| `characters_per_chunk` | [tools.py](src/react_agent/tools.py) | `1000` |
| `overlap_characters` | [tools.py](src/react_agent/tools.py) | `100` |
| Agent instructions | [prompts.py](src/react_agent/prompts.py) | — |

The Retriever runs at `temperature=0.0` (deterministic tool selection); the Generator at `temperature=0.3` with streaming enabled.

## Development

```bash
make test               # unit tests
make integration_tests  # integration tests (requires a working provider config)
make lint               # ruff + mypy
make format             # ruff format + import sort
```

## Credits

Built on top of [langchain-ai/react-agent](https://github.com/langchain-ai/react-agent), the LangGraph ReAct Agent template. The project skeleton — package layout, `langgraph.json`, `Makefile`, test setup, and the `Context`/`State` pattern — comes from there; the retrieval pipeline, vector search tool, prompts, and two-agent handoff are this project's own.

## License

MIT — see [LICENSE](LICENSE).
