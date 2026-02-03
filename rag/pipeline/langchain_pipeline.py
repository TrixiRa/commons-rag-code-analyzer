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
LangChain RAG Pipeline Implementation

Uses LangChain for building RAG chains with optional conversation memory.
"""

from typing import Any, Optional

from pipeline import BaseRAGPipeline, ConversationalRAGPipeline, RAGResult, VectorStoreProtocol
from utils.logging_config import get_logger
from utils.document_formatter import DocumentFormatter
from utils.lazy_imports import check_import_available

logger = get_logger(__name__)


def check_langchain_available() -> bool:
    """Check if LangChain is installed."""
    return check_import_available("langchain_core")


class LangChainRAGPipeline(BaseRAGPipeline):
    """
    LangChain-based RAG pipeline.
    
    Uses LangChain Expression Language (LCEL) for building chains.
    """
    
    def __init__(
        self,
        vector_store: VectorStoreProtocol,
        model: str = "tinyllama",
        provider: str = "ollama",
        top_k: int = 5
    ) -> None:
        """
        Initialize the LangChain RAG pipeline.
        
        Args:
            vector_store: Vector store for document retrieval
            model: Model name
            provider: 'ollama' or 'openai'
            top_k: Default number of documents to retrieve
        """
        if not check_langchain_available():
            raise ImportError(
                "LangChain is required for this pipeline. Install with:\n"
                "pip install langchain langchain-community langchain-ollama langchain-openai"
            )
        
        self._vector_store = vector_store
        self._model = model
        self._provider = provider
        self._top_k = top_k
        self._chain: Optional[Any] = None
        self._retriever: Optional[Any] = None
        self._formatter = DocumentFormatter()
        
        logger.info(f"Initialized LangChain pipeline with provider={provider}, model={model}")
    
    def get_provider(self) -> str:
        return self._provider
    
    def get_model(self) -> str:
        return self._model
    
    def _get_retriever(self) -> Any:
        """Get or create the LangChain retriever."""
        if self._retriever is None:
            self._retriever = HybridCodeRetriever(
                vector_store=self._vector_store,
                top_k=self._top_k
            )
        return self._retriever
    
    def _get_chain(self) -> Any:
        """Get or create the RAG chain."""
        if self._chain is None:
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.runnables import RunnablePassthrough, RunnableLambda
            from langchain_core.output_parsers import StrOutputParser
            from langchain_core.documents import Document
            
            # Get LLM
            if self._provider == "openai":
                from langchain_openai import ChatOpenAI
                llm = ChatOpenAI(model=self._model, temperature=0.1)
            else:
                from langchain_ollama import ChatOllama
                llm = ChatOllama(model=self._model, temperature=0.1)
            
            # Prompt template
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a helpful assistant that answers questions about the Apache Commons Text Java library.
You MUST base your answers strictly on the provided source code and documentation context.

IMPORTANT RULES:
1. Only answer based on the provided context. If the context doesn't contain enough information, say so clearly.
2. When referencing code, mention the specific file and class name.
3. If you're uncertain about something, express that uncertainty.
4. Do not make up information that isn't in the context."""),
                ("human", """Based on the following source code context, please answer the question.

CONTEXT:
{context}

QUESTION: {question}

Please provide a clear, accurate answer based strictly on the context above.""")
            ])
            
            retriever = self._get_retriever()
            
            def format_docs(docs: list[Any]) -> str:
                parts: list[str] = []
                for i, doc in enumerate(docs, 1):
                    header = f"=== SOURCE {i} ===\n"
                    header += f"File: {doc.metadata.get('path', 'unknown')}\n"
                    if doc.metadata.get('class_name'):
                        header += f"Class: {doc.metadata['class_name']}\n"
                    if doc.metadata.get('start_line') and doc.metadata.get('end_line'):
                        header += f"Lines: {doc.metadata['start_line']}-{doc.metadata['end_line']}\n"
                    header += f"Relevance: {doc.metadata.get('relevance_score', 0):.3f}\n"
                    header += "=" * 40 + "\n"
                    parts.append(header + doc.page_content)
                return "\n\n".join(parts)
            
            self._chain = (
                {
                    "context": retriever | RunnableLambda(format_docs),
                    "question": RunnablePassthrough()
                }
                | prompt
                | llm
                | StrOutputParser()
            )
        
        return self._chain
    
    def query(self, question: str, top_k: int = 5) -> RAGResult:
        """Answer a question using RAG."""
        try:
            chain = self._get_chain()
            logger.info(f"Querying LangChain pipeline: {question[:50]}...")
            
            answer = chain.invoke(question)
            
            # Get sources for the result
            retriever = self._get_retriever()
            docs = retriever.invoke(question)
            
            sources_parts: list[str] = []
            for doc in docs:
                source = f"- {doc.metadata.get('path', 'unknown')}"
                if doc.metadata.get('class_name'):
                    source += f" [{doc.metadata['class_name']}]"
                sources_parts.append(source)
            
            return RAGResult(
                answer=answer,
                sources="\n".join(sources_parts),
                context=None,
                uncertainty=False
            )
            
        except Exception as e:
            logger.error(f"LangChain query failed: {e}", exc_info=True)
            return RAGResult(
                answer=f"Error: {e}",
                sources="",
                uncertainty=True
            )


class LangChainConversationalPipeline(ConversationalRAGPipeline):
    """
    Conversational RAG pipeline with memory using LangChain.
    """
    
    def __init__(
        self,
        vector_store: VectorStoreProtocol,
        model: str = "tinyllama",
        provider: str = "ollama",
        top_k: int = 5
    ) -> None:
        if not check_langchain_available():
            raise ImportError("LangChain is required for this pipeline.")
        
        self._vector_store = vector_store
        self._model = model
        self._provider = provider
        self._top_k = top_k
        self._sessions: dict[str, Any] = {}
        self._chains: dict[str, Any] = {}
        
        logger.info(f"Initialized conversational pipeline with provider={provider}")
    
    def get_provider(self) -> str:
        return self._provider
    
    def get_model(self) -> str:
        return self._model
    
    def _get_session(self, session_id: str) -> Any:
        """Get or create a message history for a session."""
        if session_id not in self._sessions:
            from langchain_community.chat_message_histories import ChatMessageHistory
            self._sessions[session_id] = ChatMessageHistory()
        return self._sessions[session_id]
    
    def query(self, question: str, top_k: int = 5) -> RAGResult:
        """Answer a question (without conversation history)."""
        return self.query_with_history(question, "default", top_k)
    
    def query_with_history(
        self,
        question: str,
        session_id: str = "default",
        top_k: int = 5
    ) -> RAGResult:
        """Answer a question with conversation history."""
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
        from langchain_core.runnables import RunnablePassthrough, RunnableLambda
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.runnables.history import RunnableWithMessageHistory
        
        # Get LLM
        if self._provider == "openai":
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(model=self._model, temperature=0.1)
        else:
            from langchain_ollama import ChatOllama
            llm = ChatOllama(model=self._model, temperature=0.1)
        
        retriever = HybridCodeRetriever(
            vector_store=self._vector_store,
            top_k=top_k
        )
        
        message_history = self._get_session(session_id)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant that answers questions about the Apache Commons Text Java library.
Base your answers strictly on the provided source code context. If unsure, say so."""),
            MessagesPlaceholder(variable_name="history"),
            ("human", """Context from codebase:
{context}

Question: {input}""")
        ])
        
        def format_docs(docs: list[Any]) -> str:
            return "\n\n".join(
                f"[{doc.metadata.get('path', 'unknown')}]\n{doc.page_content[:1500]}"
                for doc in docs
            )
        
        def get_context(input_dict: dict[str, Any]) -> str:
            q = input_dict["input"]
            docs = retriever.invoke(q)
            return format_docs(docs)
        
        base_chain = (
            RunnablePassthrough.assign(context=RunnableLambda(get_context))
            | prompt
            | llm
            | StrOutputParser()
        )
        
        chain_with_history = RunnableWithMessageHistory(
            base_chain,
            lambda sid: message_history,
            input_messages_key="input",
            history_messages_key="history"
        )
        
        try:
            logger.info(f"Conversational query (session={session_id}): {question[:50]}...")
            answer = chain_with_history.invoke(
                {"input": question},
                config={"configurable": {"session_id": session_id}}
            )
            
            return RAGResult(
                answer=answer,
                sources="",
                uncertainty=False,
                metadata={"session_id": session_id}
            )
            
        except Exception as e:
            logger.error(f"Conversational query failed: {e}", exc_info=True)
            return RAGResult(
                answer=f"Error: {e}",
                sources="",
                uncertainty=True
            )
    
    def clear_history(self, session_id: str = "default") -> None:
        """Clear conversation history for a session."""
        if session_id in self._sessions:
            self._sessions[session_id].clear()
            logger.info(f"Cleared history for session: {session_id}")


# Custom retriever for LangChain integration
try:
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.documents import Document
    from langchain_core.callbacks import CallbackManagerForRetrieverRun
    from pydantic import Field
    
    class HybridCodeRetriever(BaseRetriever):
        """
        Custom LangChain retriever wrapping hybrid search logic.
        """
        
        vector_store: Any = Field(description="SimpleVectorStore instance")
        top_k: int = Field(default=5, description="Number of documents to retrieve")
        
        class Config:
            arbitrary_types_allowed = True
        
        def _get_relevant_documents(
            self,
            query: str,
            *,
            run_manager: Optional[CallbackManagerForRetrieverRun] = None
        ) -> list[Document]:
            """Retrieve relevant documents using hybrid search."""
            search_result = self.vector_store.search(query, top_k=self.top_k)
            
            docs: list[Document] = []
            for doc, score in search_result['results']:
                lc_doc = Document(
                    page_content=doc['text'],
                    metadata={
                        'path': doc.get('path', ''),
                        'type': doc.get('type', ''),
                        'class_name': doc.get('class_name'),
                        'start_line': doc.get('start_line'),
                        'end_line': doc.get('end_line'),
                        'relevance_score': score,
                        'is_folder_query': search_result.get('is_folder_query', False),
                        'folder_name': search_result.get('folder_name')
                    }
                )
                docs.append(lc_doc)
            
            return docs

except ImportError:
    class HybridCodeRetriever:  # type: ignore
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("LangChain is required for HybridCodeRetriever.")
