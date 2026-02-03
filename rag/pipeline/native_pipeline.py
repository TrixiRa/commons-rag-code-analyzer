"""
Native RAG Pipeline Implementation

Uses OpenAI-compatible API (OpenAI or Ollama) for generation.
This is the original implementation, now conforming to BaseRAGPipeline interface.
"""

import os
import re
from typing import Any, Optional

from pipeline import BaseRAGPipeline, RAGResult, VectorStoreProtocol
from utils.logging_config import get_logger
from utils.document_formatter import DocumentFormatter

logger = get_logger(__name__)


# Configuration constants
DEFAULT_LLM_PROVIDER = "ollama"
DEFAULT_OLLAMA_MODEL = "tinyllama"
DEFAULT_OLLAMA_URL = "http://localhost:11434/v1"
LOW_RELEVANCE_THRESHOLD = 0.3


class NativeRAGPipeline(BaseRAGPipeline):
    """
    Native RAG pipeline using OpenAI-compatible API.
    
    Supports:
    - Ollama (local, default): Set OLLAMA_MODEL env var to change model
    - OpenAI: Set OPENAI_API_KEY env var to use OpenAI
    """
    
    def __init__(
        self, 
        vector_store: VectorStoreProtocol, 
        provider: Optional[str] = None
    ) -> None:
        """
        Initialize the RAG pipeline.
        
        Args:
            vector_store: Vector store for document retrieval
            provider: LLM provider ('ollama' or 'openai'), auto-detected if None
        """
        self.vector_store = vector_store
        self._llm_client: Optional[Any] = None
        self._formatter = DocumentFormatter()
        
        # Auto-detect provider
        if provider:
            self._provider = provider
        elif os.environ.get("OPENAI_API_KEY"):
            self._provider = "openai"
        else:
            self._provider = "ollama"
        
        self._model = self._get_model_name()
        logger.info(f"Initialized RAG pipeline with provider={self._provider}, model={self._model}")
    
    def _get_model_name(self) -> str:
        """Get the model name based on provider."""
        if self._provider == "openai":
            return os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        return os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
    
    def _get_llm_client(self) -> Any:
        """Initialize the LLM client (lazy loading)."""
        if self._llm_client is None:
            from openai import OpenAI
            
            if self._provider == "openai":
                self._llm_client = OpenAI()
            else:
                self._llm_client = OpenAI(
                    base_url=os.environ.get("OLLAMA_URL", DEFAULT_OLLAMA_URL),
                    api_key="ollama",
                    timeout=300.0
                )
        return self._llm_client
    
    def get_provider(self) -> str:
        """Get the LLM provider name."""
        return self._provider
    
    def get_model(self) -> str:
        """Get the model name being used."""
        return self._model
    
    def query(self, question: str, top_k: int = 5) -> RAGResult:
        """
        Answer a question using RAG.
        
        Args:
            question: The question to answer
            top_k: Number of documents to retrieve
            
        Returns:
            RAGResult with answer, sources, and metadata
        """
        # Step 1: Retrieve relevant documents
        search_result = self.vector_store.search(question, top_k=top_k)
        retrieved = search_result['results']
        is_folder_query = search_result.get('is_folder_query', False)
        folder_name = search_result.get('folder_name')
        
        # Check relevance
        max_score = max((score for _, score in retrieved), default=0)
        
        if max_score < LOW_RELEVANCE_THRESHOLD:
            logger.warning(f"Low relevance scores for query: {question} (max={max_score:.2f})")
            return RAGResult(
                answer=(
                    "I could not find sufficiently relevant information in the codebase to answer "
                    "this question confidently. The retrieved documents had low relevance scores "
                    f"(max: {max_score:.2f}). Please try rephrasing your question or ask about "
                    "specific classes, methods, or concepts in the Apache Commons Text library."
                ),
                sources=self._formatter.format_sources(retrieved),
                context=None,
                uncertainty=True
            )
        
        # Step 2: Format context
        context = self._formatter.format_context(retrieved)
        
        # Step 3: Build prompt
        system_prompt = self._build_system_prompt(is_folder_query, folder_name)
        user_prompt = self._build_user_prompt(context, question)
        
        # Step 4: Call LLM
        try:
            client = self._get_llm_client()
            logger.info(f"Generating answer with {self._provider}...")
            
            response = client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            answer = response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"LLM call failed: {e}", exc_info=True)
            return self._build_error_result(e, retrieved, context)
        
        # Step 5: Check for hallucinations
        hallucinations = self._detect_hallucinations(answer, retrieved)
        
        # Build final answer
        final_answer = answer
        if hallucinations:
            warning = "\n\n⚠️ **Warning: Possible inaccuracies/hallucination detected** (post-hoc verification)\n"
            warning += "The following references could not be verified in the retrieved sources:\n"
            for h in hallucinations:
                warning += f"  - {h}\n"
            warning += "\n💡 If the correct file/class is listed in the SOURCES below, the answer still likely contains valuable information despite the mistake.\n"
            final_answer = answer + warning
        
        return RAGResult(
            answer=final_answer,
            sources=self._formatter.format_sources(retrieved),
            context=context,
            uncertainty=len(hallucinations) > 0,
            hallucinations=hallucinations,
            metadata={
                "is_folder_query": is_folder_query,
                "folder_name": folder_name,
                "max_relevance": max_score
            }
        )
    
    def _build_system_prompt(
        self, 
        is_folder_query: bool, 
        folder_name: Optional[str]
    ) -> str:
        """Build the system prompt based on query type."""
        if is_folder_query and folder_name:
            return f"""You are a helpful assistant that answers questions about the Apache Commons Text Java library.
You MUST base your answers strictly on the provided source code and documentation context.

This is a FOLDER/PACKAGE OVERVIEW question about the '{folder_name}' package.

IMPORTANT RULES FOR PACKAGE OVERVIEW:
1. Provide an overview of what this package/folder contains and its purpose.
2. List the main classes and interfaces in this package, briefly describing each one.
3. Explain the common functionality or theme that ties these classes together.
4. If there's a package-info.java file, use its description as the primary source.
5. Do NOT dive deep into one specific class - give a balanced overview of the whole package.
6. Mention how the classes relate to each other (if apparent from context).
"""
        return """You are a helpful assistant that answers questions about the Apache Commons Text Java library.
You MUST base your answers strictly on the provided source code and documentation context.

IMPORTANT RULES:
1. Only answer based on the provided context. If the context doesn't contain enough information, say so clearly.
2. When referencing code, mention the specific file and class name.
3. If you're uncertain about something, express that uncertainty.
4. Do not make up information that isn't in the context.
5. If asked about something not covered in the context, say "I don't have enough information in the retrieved context to answer this question."
"""
    
    def _build_user_prompt(self, context: str, question: str) -> str:
        """Build the user prompt with context and question."""
        return f"""Based on the following source code and documentation context, please answer the question.

CONTEXT:
{context}

QUESTION: {question}

Please provide a clear, accurate answer based strictly on the context above. Reference specific files and classes when relevant."""
    
    def _build_error_result(
        self, 
        error: Exception, 
        retrieved: list[tuple[dict[str, Any], float]], 
        context: str
    ) -> RAGResult:
        """Build an error result with helpful hints."""
        error_msg = str(error)
        if self._provider == "ollama":
            hint = (
                f"\n\nTo use Ollama:\n"
                f"  1. Install Ollama: https://ollama.ai\n"
                f"  2. Run: ollama pull {self._model}\n"
                f"  3. Start Ollama (it runs automatically on install)\n"
                f"\nOr set OPENAI_API_KEY to use OpenAI instead."
            )
        else:
            hint = "\n\nPlease check your OPENAI_API_KEY."
        
        return RAGResult(
            answer=f"Error calling LLM ({self._provider}): {error_msg}{hint}",
            sources=self._formatter.format_sources(retrieved),
            context=context,
            uncertainty=True
        )
    
    def _detect_hallucinations(
        self, 
        answer: str, 
        retrieved_docs: list[tuple[dict[str, Any], float]]
    ) -> list[str]:
        """Detect potential hallucinations in the answer."""
        hallucinations: list[str] = []
        
        # Build set of known class names
        known_classes: set[str] = set()
        known_text = ""
        
        for doc, _ in retrieved_docs:
            if doc.get('class_name'):
                known_classes.add(doc['class_name'].lower())
            known_text += doc.get('text', '') + " "
        
        for doc in self.vector_store.documents:
            if doc.get('class_name'):
                known_classes.add(doc['class_name'].lower())
            path = doc.get('path', '')
            if path.endswith('.java'):
                filename = path.split('/')[-1].replace('.java', '')
                known_classes.add(filename.lower())
        
        known_text_lower = known_text.lower()
        
        # Extract class-like names from answer
        class_pattern = r'\b([A-Z][a-zA-Z]{2,}(?:[A-Z][a-zA-Z]*)*)\b'
        mentioned_classes = set(re.findall(class_pattern, answer))
        
        # Common English words to filter out
        common_words = {
            'The', 'This', 'That', 'These', 'Those', 'When', 'Where', 'Which',
            'What', 'How', 'Why', 'For', 'From', 'With', 'About', 'Into',
            'Through', 'During', 'Before', 'After', 'Above', 'Below', 'Between',
            'Under', 'Again', 'Further', 'Then', 'Once', 'Here', 'There',
            'All', 'Each', 'Every', 'Both', 'Few', 'More', 'Most', 'Other',
            'Some', 'Such', 'Only', 'Same', 'Than', 'Very', 'Just', 'Also',
            'Now', 'Even', 'Still', 'Already', 'Always', 'Often', 'Never',
            'Java', 'String', 'Object', 'Class', 'Method', 'Example', 'Code',
            'File', 'System', 'Output', 'Input', 'Error', 'Exception', 'True',
            'False', 'Null', 'Return', 'Public', 'Private', 'Static', 'Final',
            'Apache', 'Commons', 'Text', 'Library', 'Based', 'Using', 'Called',
            'Following', 'Specific', 'Default', 'Custom', 'Similar', 'Different',
            'Overall', 'Simply', 'Powerful', 'Provides', 'Returns', 'Takes',
            'Modified', 'Given', 'Corresponding', 'Variable', 'Variables',
            'Values', 'Customized', 'Efficient', 'Replaced', 'Replacing',
            'Working', 'Convenience', 'Regular', 'Parameters', 'Constructor',
            'Prefix', 'Suffix', 'Definition', 'Instead', 'However', 'Therefore'
        }
        
        for class_name in mentioned_classes:
            if class_name in common_words:
                continue
            
            class_lower = class_name.lower()
            found = False
            
            for known in known_classes:
                if class_lower == known or class_lower in known or known in class_lower:
                    found = True
                    break
            
            if not found and class_lower in known_text_lower:
                found = True
            
            if not found:
                hallucinations.append(f"Class '{class_name}' not found in codebase")
        
        return hallucinations[:5]


# Backwards compatibility alias
RAGPipeline = NativeRAGPipeline
