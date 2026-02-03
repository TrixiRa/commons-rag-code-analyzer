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
Document Formatting Utilities

Provides consistent document formatting for RAG context building.
Used by both native and LangChain RAG pipeline implementations.
"""

from typing import Any, Protocol, Optional
from dataclasses import dataclass


@dataclass
class FormattedDocument:
    """A formatted document with metadata."""
    content: str
    path: str
    doc_type: str
    class_name: Optional[str] = None
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    relevance_score: float = 0.0


class DocumentLike(Protocol):
    """Protocol for document-like objects (duck typing)."""
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a value by key."""
        ...


class DocumentFormatter:
    """
    Formats documents for RAG context and source references.
    
    Provides consistent formatting across different RAG implementations
    (native pipeline and LangChain integration).
    """
    
    def __init__(
        self,
        max_chars_per_doc: int = 2000,
        show_line_numbers: bool = True,
        show_relevance: bool = True
    ) -> None:
        """
        Initialize the formatter.
        
        Args:
            max_chars_per_doc: Maximum characters per document (for LLM context limits)
            show_line_numbers: Include line number information
            show_relevance: Include relevance scores
        """
        self.max_chars_per_doc = max_chars_per_doc
        self.show_line_numbers = show_line_numbers
        self.show_relevance = show_relevance
    
    def format_context(
        self,
        documents: list[tuple[dict[str, Any], float]],
        header_template: str = "=== SOURCE {index} ==="
    ) -> str:
        """
        Format retrieved documents into a context string for LLM.
        
        Args:
            documents: List of (document_dict, score) tuples
            header_template: Template for document headers
            
        Returns:
            Formatted context string
        """
        context_parts: list[str] = []
        
        for i, (doc, score) in enumerate(documents, 1):
            header_lines = [header_template.format(index=i)]
            header_lines.append(f"File: {doc.get('path', 'unknown')}")
            header_lines.append(f"Type: {doc.get('type', 'unknown')}")
            
            if doc.get('class_name'):
                header_lines.append(f"Class: {doc['class_name']}")
            
            if self.show_line_numbers:
                start_line = doc.get('start_line')
                end_line = doc.get('end_line')
                if start_line and end_line:
                    header_lines.append(f"Lines: {start_line}-{end_line}")
            
            if self.show_relevance:
                header_lines.append(f"Relevance: {score:.3f}")
            
            header_lines.append("=" * 40)
            header = "\n".join(header_lines)
            
            # Truncate long documents
            text = doc.get('text', '')
            if len(text) > self.max_chars_per_doc:
                text = text[:self.max_chars_per_doc] + "\n... [truncated for context length]"
            
            context_parts.append(header + "\n" + text)
        
        return "\n\n".join(context_parts)
    
    def format_sources(
        self,
        documents: list[tuple[dict[str, Any], float]]
    ) -> str:
        """
        Format source references for display to user.
        
        Args:
            documents: List of (document_dict, score) tuples
            
        Returns:
            Formatted sources string
        """
        sources: list[str] = []
        
        for doc, score in documents:
            source = f"- {doc.get('path', 'unknown')}"
            
            # Add line numbers if available
            if self.show_line_numbers:
                start_line = doc.get('start_line')
                end_line = doc.get('end_line')
                if start_line and end_line:
                    if start_line == end_line:
                        source += f" (L{start_line})"
                    else:
                        source += f" (L{start_line}-{end_line})"
            
            if doc.get('class_name'):
                source += f" [{doc['class_name']}]"
            
            if self.show_relevance:
                source += f" [relevance: {score:.2f}]"
            
            sources.append(source)
        
        return "\n".join(sources)
    
    def format_for_langchain(
        self,
        documents: list[tuple[dict[str, Any], float]],
        max_chars: int = 1500
    ) -> str:
        """
        Format documents for LangChain chains (more compact format).
        
        Args:
            documents: List of (document_dict, score) tuples
            max_chars: Maximum characters per document
            
        Returns:
            Compact formatted string
        """
        parts: list[str] = []
        
        for doc, score in documents:
            path = doc.get('path', 'unknown')
            text = doc.get('text', '')[:max_chars]
            parts.append(f"[{path}]\n{text}")
        
        return "\n\n".join(parts)


# Default formatter instance
_default_formatter: Optional[DocumentFormatter] = None


def get_default_formatter() -> DocumentFormatter:
    """Get the default document formatter instance."""
    global _default_formatter
    if _default_formatter is None:
        _default_formatter = DocumentFormatter()
    return _default_formatter
