"""Code RAG Web Server."""

import argparse
import os

import uvicorn


def main():
    """Entry point for rag-server command."""
    parser = argparse.ArgumentParser(description="Code RAG Web Server")
    parser.add_argument("--config", type=str, default=None, help="YAML 配置文件路径")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")), help="监听端口")
    args = parser.parse_args()

    # Pass config path to the app via environment variable
    if args.config:
        os.environ["RAG_CONFIG"] = args.config

    uvicorn.run("server.app:app", host=args.host, port=args.port, reload=True)
