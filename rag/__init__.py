"""
RAG Pipeline for Apache Commons Text

A Retrieval Augmented Generation (RAG) system for answering questions
about the Apache Commons Text Java library.

Main Entry Points:
    - main.py: CLI interface
    - pipeline/: RAG pipeline implementations
    - analysis/: Architecture analysis tools
    - utils/: Shared utilities
"""

from config import REPO_ROOT, INDEX_DIR, DATA_DIR

__version__ = "2.0.0"
__all__ = ["REPO_ROOT", "INDEX_DIR", "DATA_DIR"]
