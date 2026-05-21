"""FastAPI application for code-rag web UI."""

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from rag.config import ConfigManager, RAGConfig
from rag.rag_system import RAGSystem

app = FastAPI(title="Code RAG")

# Static files
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Allowed docs directory (only serve files under sample-docs)
DOCS_ROOT = Path(os.environ.get("RAG_DOCS_ROOT", "sample-docs")).resolve()

# Load config once at startup
_config: RAGConfig | None = None
_system: RAGSystem | None = None


def get_config() -> RAGConfig:
    global _config
    if _config is None:
        manager = ConfigManager()
        config_path = os.environ.get("RAG_CONFIG", "rag_config.yaml")
        if Path(config_path).exists():
            _config = manager.load_config(config_path)
        else:
            _config = manager.load_config(None)
    return _config


async def get_system() -> RAGSystem:
    global _system
    if _system is None:
        config = get_config()
        _system = RAGSystem(config)
        await _system.__aenter__()
        _system.bm25_search.ensure_index()
    return _system


class QueryRequest(BaseModel):
    query: str
    top_n: int = 5


class ChunkResult(BaseModel):
    chunk_id: str
    content: str
    source_file: str
    chunk_type: str
    heading_text: str
    start_line: int
    score: float
    dense_score: float | None = None
    bm25_score: float | None = None


class QueryResponse(BaseModel):
    chunks: list[ChunkResult]
    rewritten_query: str | None = None


@app.get("/")
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.post("/api/query")
async def query(req: QueryRequest) -> QueryResponse:
    system = await get_system()
    detailed = await system.retrieve_detailed(req.query, top_n=req.top_n)

    rewritten_query = detailed["rewritten_query"]
    chunks = []
    for r in detailed["results"]:
        chunk = r["chunk"]
        content = (
            f"{chunk.context}\n\n{chunk.content}" if chunk.context else chunk.content
        )
        chunks.append(
            ChunkResult(
                chunk_id=chunk.chunk_id,
                content=content,
                source_file=chunk.source_file,
                chunk_type=chunk.chunk_type.value,
                heading_text=chunk.heading_text,
                start_line=chunk.start_line,
                score=round(r["score"], 6),
                dense_score=round(r["dense_score"], 6) if r["dense_score"] is not None else None,
                bm25_score=round(r["bm25_score"], 6) if r["bm25_score"] is not None else None,
            )
        )

    return QueryResponse(
        chunks=chunks,
        rewritten_query=rewritten_query if rewritten_query != req.query else None,
    )


@app.get("/api/file")
async def get_file(path: str) -> PlainTextResponse:
    """Return markdown file content. Only serves files under DOCS_ROOT."""
    # Resolve and validate path is under docs root
    file_path = (DOCS_ROOT / path).resolve()
    if not str(file_path).startswith(str(DOCS_ROOT)):
        raise HTTPException(status_code=403, detail="Access denied")
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    if file_path.suffix not in (".md", ".txt"):
        raise HTTPException(status_code=403, detail="Only markdown files allowed")

    content = file_path.read_text(encoding="utf-8")
    return PlainTextResponse(content)
