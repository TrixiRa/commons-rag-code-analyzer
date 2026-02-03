# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
LangChain Integration Module

This module provides backwards compatibility with the old langchain_integration.py.
The actual implementation is now in pipeline/langchain_pipeline.py.
"""

from typing import Any, Optional

from pipeline.langchain_pipeline import (
    LangChainRAGPipeline,
    LangChainConversationalPipeline,
    HybridCodeRetriever,
    check_langchain_available,
)
from vector_store import SimpleVectorStore


def get_langchain_retriever(vector_store: SimpleVectorStore, top_k: int = 5) -> Any:
    """
    Create a LangChain-compatible retriever from our vector store.
    
    Args:
        vector_store: Initialized SimpleVectorStore with loaded index
        top_k: Number of documents to retrieve
        
    Returns:
        HybridCodeRetriever instance
        
    Raises:
        ImportError: If LangChain is not installed
    """
    if not check_langchain_available():
        raise ImportError(
            "LangChain is required for this feature. Install with:\n"
            "pip install langchain langchain-community langchain-ollama"
        )
    
    return HybridCodeRetriever(vector_store=vector_store, top_k=top_k)


def create_rag_chain(
    vector_store: SimpleVectorStore,
    model: str = "tinyllama",
    provider: str = "ollama",
    top_k: int = 5
) -> LangChainRAGPipeline:
    """
    Create a LangChain RAG pipeline using our hybrid retriever.
    
    Args:
        vector_store: Initialized SimpleVectorStore with loaded index
        model: Model name (default: tinyllama)
        provider: 'ollama' or 'openai'
        top_k: Number of documents to retrieve
        
    Returns:
        LangChainRAGPipeline instance
    """
    return LangChainRAGPipeline(
        vector_store=vector_store,
        model=model,
        provider=provider,
        top_k=top_k
    )


def create_conversational_chain(
    vector_store: SimpleVectorStore,
    model: str = "tinyllama",
    provider: str = "ollama",
    top_k: int = 5
) -> tuple[LangChainConversationalPipeline, Any]:
    """
    Create a conversational RAG chain with memory.
    
    Args:
        vector_store: Initialized SimpleVectorStore with loaded index
        model: Model name (default: tinyllama)
        provider: 'ollama' or 'openai'
        top_k: Number of documents to retrieve
        
    Returns:
        Tuple of (pipeline, memory) for backwards compatibility
    """
    pipeline = LangChainConversationalPipeline(
        vector_store=vector_store,
        model=model,
        provider=provider,
        top_k=top_k
    )
    # Return pipeline twice for backwards compatibility (old code expected tuple)
    return pipeline, pipeline._sessions


def quick_query(question: str, model: str = "tinyllama") -> str:
    """
    Quick one-shot query using LangChain.
    
    Args:
        question: The question to answer
        model: Ollama model to use
        
    Returns:
        The answer string
    """
    vs = SimpleVectorStore()
    vs.load_index()
    pipeline = create_rag_chain(vs, model=model)
    result = pipeline.query(question)
    return result.answer


__all__ = [
    "get_langchain_retriever",
    "create_rag_chain",
    "create_conversational_chain",
    "quick_query",
    "check_langchain_available",
    "HybridCodeRetriever",
    "LangChainRAGPipeline",
    "LangChainConversationalPipeline",
]
