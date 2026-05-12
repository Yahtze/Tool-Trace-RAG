#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tool_trace_rag.config import QUERY_TOP_K
from tool_trace_rag.memory.vector_store import DEFAULT_COLLECTION_NAME, DEFAULT_VECTOR_DIR, TraceVectorStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Query persisted trace vectors without running the agent.")
    parser.add_argument("task", help="Natural-language task string to search for.")
    parser.add_argument("--top-k", type=int, default=QUERY_TOP_K, help="Number of ranked trace results to return.")
    parser.add_argument("--vector-dir", default=str(DEFAULT_VECTOR_DIR), help="Directory for the local Chroma vector store.")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION_NAME, help="Chroma collection name.")
    args = parser.parse_args()

    store = TraceVectorStore(vector_dir=args.vector_dir, collection_name=args.collection)
    results = store.query(args.task, top_k=args.top_k)
    if not results:
        print("No trace results found.")
        return
    for result in results:
        metadata = result.metadata
        print(f"{result.rank}. trace_id={metadata.get('trace_id', '')} task_id={metadata.get('task_id', '')} score={result.score:.6f}")
        print(f"   source={metadata.get('relative_source_path') or metadata.get('source_path', '')}")
        print(f"   task={metadata.get('task_preview', '')}")


if __name__ == "__main__":
    main()
