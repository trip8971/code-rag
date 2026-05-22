# code-rag

Hybrid retrieval (BM25 + dense + reranker) for code documentation.  
Demo: [https://code-rag-2fp7.onrender.com/](https://code-rag-2fp7.onrender.com/)

## Install

```bash
# Core dependencies + editable install of this package
uv sync

# Optional: local BGE embeddings via sentence-transformers (~2GB)
uv sync --extra bge

# Optional: dev / test dependencies
uv sync --extra dev

# Everything
uv sync --extra bge --extra dev
```

`uv sync` creates `.venv/` automatically and maintains `uv.lock` for reproducible installs.
Use `uv sync --locked` in CI to enforce the lock file.

## Usage

```bash
# Index a directory of markdown docs
uv run rag --config rag_config.yaml index path/to/docs/

# Run a query
uv run rag --config rag_config.yaml query "example code of useEffect"
```

Or activate the venv and call `rag` directly:

```bash
source .venv/bin/activate
rag --config rag_config.yaml query "..."
```

See `rag_config.yaml` for service configuration (embedding / reranker / query rewriter).
