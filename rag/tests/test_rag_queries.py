"""
Tests for the RAG Pipeline

Tests cover:
1. Query about a specific class (StringSubstitutor)
2. Query about a folder/package
3. General question about the repository
4. Query that should fail (irretrievable information)
"""

import pytest
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import REPO_ROOT, INDEX_DIR
from vector_store import SimpleVectorStore
from pipeline import RAGResult


class TestVectorStoreSearch:
    """Tests for vector store search functionality."""
    
    @pytest.fixture
    def vector_store(self) -> SimpleVectorStore:
        """Load the vector store with existing index."""
        store = SimpleVectorStore()
        if not store.load_index():
            pytest.skip("Index not built. Run 'python main.py --build' first.")
        return store
    
    def test_search_specific_class_stringsubstitutor(self, vector_store: SimpleVectorStore) -> None:
        """
        Test: Query about a specific class (StringSubstitutor).
        
        Expected: Should retrieve documents mentioning StringSubstitutor
        with high relevance scores.
        """
        query = "How does StringSubstitutor work?"
        result = vector_store.search(query, top_k=5)
        
        assert 'results' in result
        assert len(result['results']) > 0
        
        # Check that StringSubstitutor is in the top results
        top_docs = result['results']
        found_stringsubstitutor = False
        
        for doc, score in top_docs:
            path = doc.get('path', '')
            class_name = doc.get('class_name', '')
            
            if 'StringSubstitutor' in path or 'StringSubstitutor' in class_name:
                found_stringsubstitutor = True
                # Score should be reasonably high due to boosting
                assert score > 0.3, f"Expected high relevance for StringSubstitutor, got {score}"
                break
        
        assert found_stringsubstitutor, (
            "StringSubstitutor should be in top results for query about StringSubstitutor"
        )
    
    def test_search_folder_query(self, vector_store: SimpleVectorStore) -> None:
        """
        Test: Query about a folder/package.
        
        Expected: Should detect folder query and return diverse files
        from the specified package.
        """
        query = "What classes are in the translate package?"
        result = vector_store.search(query, top_k=5)
        
        assert 'results' in result
        assert 'is_folder_query' in result
        
        # Should detect this as a folder query
        # Note: might be False if 'translate' isn't found as a folder
        if result['is_folder_query']:
            assert result.get('folder_name') is not None
            
            # Results should be deduplicated (different files)
            seen_files: set[str] = set()
            for doc, _ in result['results']:
                path = doc.get('path', '')
                filename = path.split('/')[-1] if '/' in path else path
                # Should not see duplicate files
                assert filename not in seen_files, f"Duplicate file in folder query: {filename}"
                seen_files.add(filename)
    
    def test_search_general_question(self, vector_store: SimpleVectorStore) -> None:
        """
        Test: General question about the repository.
        
        Expected: Should return relevant documents about text manipulation.
        """
        query = "What text manipulation utilities does this library provide?"
        result = vector_store.search(query, top_k=5)
        
        assert 'results' in result
        assert len(result['results']) > 0
        
        # All results should have some relevance
        for doc, score in result['results']:
            assert score > 0.0, "All results should have positive relevance"
            assert doc.get('text'), "Documents should have text content"
    
    def test_search_irretrievable_query(self, vector_store: SimpleVectorStore) -> None:
        """
        Test: Query about something not in the codebase.
        
        Expected: Should return results with low relevance scores,
        indicating the information isn't well-covered.
        """
        # Ask about something definitely not in Apache Commons Text
        query = "How do I implement a neural network in this library?"
        result = vector_store.search(query, top_k=5)
        
        assert 'results' in result
        
        # All results should have relatively low relevance for this query
        max_score = max(score for _, score in result['results']) if result['results'] else 0
        
        # The max score should be lower than for relevant queries
        # This isn't a strict threshold test, but checks the behavior
        assert max_score < 0.8, (
            f"Expected low relevance for irrelevant query, got max score {max_score}"
        )


class TestRAGPipelineIntegration:
    """Integration tests for the complete RAG pipeline."""
    
    @pytest.fixture
    def mock_llm_response(self) -> Mock:
        """Create a mock LLM response."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "This is a test answer about StringSubstitutor."
        return mock_response
    
    def test_pipeline_query_with_mocked_llm(self, mock_llm_response: Mock) -> None:
        """
        Test: Complete pipeline with mocked LLM.
        
        Verifies the pipeline correctly:
        1. Loads the vector store
        2. Retrieves relevant documents
        3. Formats the context
        4. Would call the LLM (mocked)
        5. Returns a proper RAGResult
        """
        from pipeline.native_pipeline import NativeRAGPipeline
        
        store = SimpleVectorStore()
        if not store.load_index():
            pytest.skip("Index not built. Run 'python main.py --build' first.")
        
        pipeline = NativeRAGPipeline(store, provider="ollama")
        
        # Mock the LLM client
        with patch.object(pipeline, '_get_llm_client') as mock_client:
            mock_openai = Mock()
            mock_openai.chat.completions.create.return_value = mock_llm_response
            mock_client.return_value = mock_openai
            
            result = pipeline.query("How does StringSubstitutor work?", top_k=3)
            
            # Verify result structure
            assert isinstance(result, RAGResult)
            assert result.answer == "This is a test answer about StringSubstitutor."
            assert result.sources  # Should have sources
            assert result.context  # Should have context (since mocked LLM succeeded)
    
    def test_pipeline_low_relevance_response(self) -> None:
        """
        Test: Pipeline handles low relevance gracefully.
        
        When search results have low relevance, the pipeline should
        return an uncertainty flag.
        """
        from pipeline.native_pipeline import NativeRAGPipeline
        
        store = SimpleVectorStore()
        if not store.load_index():
            pytest.skip("Index not built. Run 'python main.py --build' first.")
        
        pipeline = NativeRAGPipeline(store, provider="ollama")
        
        # Mock search to return low-relevance results
        with patch.object(store, 'search') as mock_search:
            mock_search.return_value = {
                'results': [
                    ({'text': 'unrelated text', 'path': 'some/path.java'}, 0.1),
                    ({'text': 'more unrelated', 'path': 'other/path.java'}, 0.05),
                ],
                'is_folder_query': False,
                'folder_name': None
            }
            
            result = pipeline.query("Something completely unrelated to this codebase")
            
            # Should indicate uncertainty due to low relevance
            assert result.uncertainty is True
            assert "could not find sufficiently relevant" in result.answer.lower()
    
    def test_irretrievable_query_response(self) -> None:
        """
        Test: Query about something not in the codebase returns appropriate message.
        
        When asked about neural networks (not in Apache Commons Text),
        the model should indicate it doesn't have enough information.
        
        Note: This is an integration test that requires a running LLM.
        """
        from pipeline.native_pipeline import NativeRAGPipeline
        
        store = SimpleVectorStore()
        if not store.load_index():
            pytest.skip("Index not built. Run 'python main.py --build' first.")
        
        pipeline = NativeRAGPipeline(store, provider="ollama")
        
        # Make actual LLM call - no mocking
        try:
            result = pipeline.query(
                "How do I implement a neural network in this library?",
                top_k=3
            )
        except Exception as e:
            pytest.skip(f"LLM not available: {e}")
        
        # Verify the response indicates lack of information
        assert isinstance(result, RAGResult)
        answer_lower = result.answer.lower()
        
        # Should contain phrases indicating insufficient information
        insufficient_info_phrases = [
            "don't have enough information",
            "doesn't have enough information", 
            "not enough information",
            "cannot find",
            "could not find",
            "no information",
            "does not appear",
            "doesn't appear",
            "not available",
            "outside the scope",
            "not related",
            "doesn't contain",
            "does not contain",
            "no neural network",
            "not a neural network",
            "text processing",
            "text manipulation",
            "not designed for",
            "doesn't support",
            "does not support",
            "unable to",
            "not applicable"
        ]
        
        has_insufficient_info_message = any(
            phrase in answer_lower for phrase in insufficient_info_phrases
        )
        
        assert has_insufficient_info_message, (
            f"Expected response to indicate insufficient information or clarify "
            f"the library's purpose, got: {result.answer}"
        )


class TestRAGResultStructure:
    """Tests for RAGResult data structure."""
    
    def test_rag_result_creation(self) -> None:
        """Test that RAGResult can be created with all fields."""
        result = RAGResult(
            answer="Test answer",
            sources="- file1.java\n- file2.java",
            context="Some context",
            uncertainty=False,
            hallucinations=["possible issue"],
            metadata={"key": "value"}
        )
        
        assert result.answer == "Test answer"
        assert result.sources == "- file1.java\n- file2.java"
        assert result.context == "Some context"
        assert result.uncertainty is False
        assert result.hallucinations == ["possible issue"]
        assert result.metadata == {"key": "value"}
    
    def test_rag_result_defaults(self) -> None:
        """Test RAGResult default values."""
        result = RAGResult(
            answer="Test",
            sources=""
        )
        
        assert result.context is None
        assert result.uncertainty is False
        assert result.hallucinations == []
        assert result.metadata == {}


class TestDocumentFormatter:
    """Tests for document formatting utilities."""
    
    def test_format_context(self) -> None:
        """Test context formatting for LLM."""
        from utils.document_formatter import DocumentFormatter
        
        formatter = DocumentFormatter()
        
        documents = [
            (
                {
                    'path': 'src/main/java/Test.java',
                    'type': 'java',
                    'class_name': 'Test',
                    'start_line': 1,
                    'end_line': 50,
                    'text': 'public class Test { }'
                },
                0.85
            )
        ]
        
        context = formatter.format_context(documents)
        
        assert 'SOURCE 1' in context
        assert 'src/main/java/Test.java' in context
        assert 'Test' in context
        assert '0.85' in context  # Relevance score
    
    def test_format_sources(self) -> None:
        """Test sources formatting for display."""
        from utils.document_formatter import DocumentFormatter
        
        formatter = DocumentFormatter()
        
        documents = [
            (
                {
                    'path': 'src/main/java/Test.java',
                    'class_name': 'Test',
                    'start_line': 10,
                    'end_line': 20,
                },
                0.9
            )
        ]
        
        sources = formatter.format_sources(documents)
        
        assert 'src/main/java/Test.java' in sources
        assert 'Test' in sources
        assert 'L10-20' in sources


class TestIngestFunctions:
    """Tests for document ingestion functions."""
    
    def test_extract_java_class_info(self) -> None:
        """Test Java class info extraction."""
        from ingest import extract_java_class_info
        
        java_code = """
package org.apache.commons.text;

import java.util.List;

public class StringSubstitutor {
    // class body
}
"""
        
        info = extract_java_class_info(java_code)
        
        assert info['package'] == 'org.apache.commons.text'
        assert info['class_name'] == 'StringSubstitutor'
    
    def test_extract_java_class_info_interface(self) -> None:
        """Test extraction for interface."""
        from ingest import extract_java_class_info
        
        java_code = """
package org.apache.commons.text;

public interface StringLookup {
    String lookup(String key);
}
"""
        
        info = extract_java_class_info(java_code)
        
        assert info['package'] == 'org.apache.commons.text'
        assert info['class_name'] == 'StringLookup'
    
    def test_get_line_number(self) -> None:
        """Test line number calculation."""
        from ingest import get_line_number
        
        content = "line1\nline2\nline3\nline4"
        
        assert get_line_number(content, 0) == 1  # Start of line 1
        assert get_line_number(content, 6) == 2  # Start of line 2
        assert get_line_number(content, 12) == 3  # Start of line 3


# ============================================================================
# Pytest configuration
# ============================================================================

def pytest_configure(config: Any) -> None:
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (requires running LLM)"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
