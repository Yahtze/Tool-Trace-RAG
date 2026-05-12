#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tool_trace_rag.memory.vector_store import DEFAULT_COLLECTION_NAME, DEFAULT_VECTOR_DIR, TraceVectorStore
from tool_trace_rag.traces.store import DEFAULT_TRACE_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description="Index persisted agent traces into a local vector store.")
    parser.add_argument("--trace-dir", default=str(DEFAULT_TRACE_DIR), help="Directory containing persisted trace JSON files.")
    parser.add_argument("--vector-dir", default=str(DEFAULT_VECTOR_DIR), help="Directory for the local Chroma vector store.")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION_NAME, help="Chroma collection name.")
    parser.add_argument("--reindex", action="store_true", help="Upsert existing trace vector records instead of skipping duplicates.")
    parser.add_argument("--clear", action="store_true", help="Clear the collection before indexing.")
    args = parser.parse_args()

    store = TraceVectorStore(vector_dir=args.vector_dir, collection_name=args.collection)
    if args.clear:
        store.clear()
    summary = store.index_directory(args.trace_dir, reindex=args.reindex)

    print(f"Indexed traces: {summary.indexed_traces}")
    print(f"Skipped duplicates: {summary.skipped_duplicates}")
    print(f"Failed traces: {summary.failed_traces}")
    print(f"Vector directory: {args.vector_dir}")
    print(f"Collection: {args.collection}")
    for error in summary.errors:
        print(f"Error: {error}")


if __name__ == "__main__":
    main()
