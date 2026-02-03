"""
Minimal RAG Pipeline for Apache Commons Text Codebase
======================================================

A minimal Retrieval Augmented Generation (RAG) system for answering questions
about the Apache Commons Text Java library, with architecture analysis capabilities.

Usage:
    python main.py --build              # Build the vector index
    python main.py --query "question"   # Ask a question
    python main.py --interactive        # Interactive Q&A mode
    python main.py --analyze            # Run architecture analysis
"""

import argparse
import sys
from typing import Optional

from config import REPO_ROOT
from utils.logging_config import get_logger, setup_logging
from utils.lazy_imports import LazyClassLoader

logger = get_logger(__name__)

# Lazy loaders for heavy modules
_SimpleVectorStore = LazyClassLoader('vector_store', 'SimpleVectorStore')
_NativeRAGPipeline = LazyClassLoader('pipeline.native_pipeline', 'NativeRAGPipeline')
_analyze_architecture = None


def _get_analyze_architecture():
    """Lazy load the architecture analysis function."""
    global _analyze_architecture
    if _analyze_architecture is None:
        from architecture_agent import analyze_architecture
        _analyze_architecture = analyze_architecture
    return _analyze_architecture


# ============================================================================
# MAIN ENTRY POINTS
# ============================================================================

def build_index(
    repo_root=REPO_ROOT,
    max_java_files: int = -1,
    max_test_files: int = -1,
    max_workers: int = 4,
    batch_size: int = 32
) -> 'SimpleVectorStore':
    """
    Build the vector index from the repository.
    
    Args:
        repo_root: Path to repository root
        max_java_files: Max Java files to process (-1 for all)
        max_test_files: Max test files to process (-1 for all)
        max_workers: Parallel workers for ingestion (1 for sequential)
        batch_size: Embedding batch size (lower = less memory)
        
    Returns:
        Initialized SimpleVectorStore
    """
    from ingest import ingest
    from vector_store import SimpleVectorStore
    
    logger.info("=" * 60)
    logger.info("RAG INDEX BUILD")
    logger.info("=" * 60)
    logger.info(f"Repository: {repo_root}")
    logger.info(f"Workers: {max_workers}, Batch size: {batch_size}")
    
    logger.info("Step 1/2: Ingesting documents...")
    parallel = max_workers > 1
    documents = ingest(
        repo_root,
        parallel=parallel,
        max_java_files=max_java_files,
        max_test_files=max_test_files,
        max_workers=max_workers
    )
    
    logger.info("Step 2/2: Building vector index...")
    store = SimpleVectorStore()
    store.build_index(documents, batch_size=batch_size)
    return store


def create_rag_pipeline(provider: Optional[str] = None) -> 'NativeRAGPipeline':
    """
    Create a RAG pipeline, loading or building the index as needed.
    
    Args:
        provider: LLM provider ('ollama' or 'openai'), auto-detected if None
        
    Returns:
        Initialized RAG pipeline
    """
    from vector_store import SimpleVectorStore
    from pipeline.native_pipeline import NativeRAGPipeline
    
    store = SimpleVectorStore()
    
    if not store.load_index():
        logger.info("No existing index found. Building new index...")
        store = build_index()
    
    return NativeRAGPipeline(store, provider=provider)


def ask(question: str, top_k: int = 5) -> None:
    """
    Convenience function to ask a question and print the answer.
    
    Args:
        question: Question to ask
        top_k: Number of documents to retrieve
    """
    pipeline = create_rag_pipeline()
    result = pipeline.query(question, top_k=top_k)
    
    print("\n" + "=" * 60)
    print("ANSWER:")
    print("=" * 60)
    print(result.answer)
    
    print("\n" + "-" * 60)
    print("SOURCES USED:")
    print("-" * 60)
    print(result.sources)
    
    if result.uncertainty:
        print("\n⚠️  Note: This answer may be incomplete due to low relevance or errors.")


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main() -> None:
    """Main CLI entry point."""
    setup_logging()
    
    parser = argparse.ArgumentParser(
        description="RAG Pipeline for Apache Commons Text",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --build
  python main.py --query "How does StringSubstitutor work?"
  python main.py --interactive
  python main.py --analyze
        """
    )
    parser.add_argument(
        "--build", action="store_true",
        help="Build/rebuild the vector index"
    )
    parser.add_argument(
        "--query", "-q", type=str,
        help="Ask a question about the codebase"
    )
    parser.add_argument(
        "--top-k", "-k", type=int, default=5,
        help="Number of documents to retrieve (default: 5)"
    )
    parser.add_argument(
        "--interactive", "-i", action="store_true",
        help="Start interactive Q&A mode"
    )
    parser.add_argument(
        "--analyze", "-a", action="store_true",
        help="Run architecture analysis"
    )
    parser.add_argument(
        "--safe", "-s", action="store_true",
        help="Safe mode: low memory usage (sequential processing, small batches)"
    )
    parser.add_argument(
        "--workers", "-w", type=int, default=4,
        help="Number of parallel workers for ingestion (default: 4)"
    )
    parser.add_argument(
        "--provider", "-p", type=str, choices=["ollama", "openai"],
        help="LLM provider (default: auto-detect)"
    )
    
    args = parser.parse_args()
    
    if args.build:
        if args.safe:
            logger.info("🛡️  SAFE MODE: Using minimal memory configuration")
            logger.info("   - Sequential processing (1 worker)")
            logger.info("   - Small batch sizes for embeddings (8)")
            build_index(max_workers=1, batch_size=8)
        else:
            build_index(max_workers=args.workers)
        logger.info("Index built successfully!")
    
    elif args.query:
        ask(args.query, top_k=args.top_k)
    
    elif args.interactive:
        pipeline = create_rag_pipeline(provider=args.provider)
        print("\nRAG Pipeline ready. Type 'quit' to exit.\n")
        
        while True:
            try:
                question = input("Question: ").strip()
                if question.lower() in ['quit', 'exit', 'q']:
                    break
                if not question:
                    continue
                
                result = pipeline.query(question, top_k=args.top_k)
                
                print("\n" + "=" * 60)
                print("ANSWER:")
                print("=" * 60)
                print(result.answer)
                
                print("\n" + "-" * 60)
                print("SOURCES USED:")
                print("-" * 60)
                print(result.sources)
                print()
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
    
    elif args.analyze:
        analyze_architecture = _get_analyze_architecture()
        analyze_architecture(REPO_ROOT)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
