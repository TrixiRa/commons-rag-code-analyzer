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
RAG Pipeline Interfaces and Base Classes

Defines the abstract interface for RAG pipelines, enabling different
implementations (native, LangChain) to be used interchangeably.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol


@dataclass
class RAGResult:
    """Result of a RAG query."""
    answer: str
    sources: str
    context: Optional[str] = None
    uncertainty: bool = False
    hallucinations: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class VectorStoreProtocol(Protocol):
    """Protocol defining the vector store interface."""
    
    def search(self, query: str, top_k: int = 5) -> dict[str, Any]:
        """Search for relevant documents."""
        ...
    
    def load_index(self) -> bool:
        """Load the index from disk."""
        ...
    
    def build_index(self, documents: list[dict[str, Any]], batch_size: int = 32) -> None:
        """Build the index from documents."""
        ...
    
    @property
    def documents(self) -> list[dict[str, Any]]:
        """All indexed documents."""
        ...


class BaseRAGPipeline(ABC):
    """
    Abstract base class for RAG pipelines.
    
    Defines the interface that all RAG implementations must follow,
    enabling dependency injection and easy swapping of implementations.
    """
    
    @abstractmethod
    def query(self, question: str, top_k: int = 5) -> RAGResult:
        """
        Answer a question using RAG.
        
        Args:
            question: The question to answer
            top_k: Number of documents to retrieve
            
        Returns:
            RAGResult with answer, sources, and metadata
        """
        pass
    
    @abstractmethod
    def get_provider(self) -> str:
        """Get the LLM provider name (e.g., 'ollama', 'openai')."""
        pass
    
    @abstractmethod
    def get_model(self) -> str:
        """Get the model name being used."""
        pass


class ConversationalRAGPipeline(BaseRAGPipeline):
    """Extended interface for conversational RAG with memory."""
    
    @abstractmethod
    def query_with_history(
        self, 
        question: str, 
        session_id: str = "default",
        top_k: int = 5
    ) -> RAGResult:
        """
        Answer a question with conversation history.
        
        Args:
            question: The question to answer
            session_id: Session identifier for conversation tracking
            top_k: Number of documents to retrieve
            
        Returns:
            RAGResult with answer considering conversation history
        """
        pass
    
    @abstractmethod
    def clear_history(self, session_id: str = "default") -> None:
        """Clear conversation history for a session."""
        pass
