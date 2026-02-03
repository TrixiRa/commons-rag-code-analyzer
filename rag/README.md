# RAG Pipeline for Apache Commons Text

A minimal Retrieval Augmented Generation (RAG) system for answering questions about the Apache Commons Text Java library, with architecture analysis capabilities.

## Attribution

This repository contains the [Apache Commons Text](https://commons.apache.org/proper/commons-text/) library source code, licensed under the [Apache License 2.0](../LICENSE.txt).

The `rag/` directory contains original code for RAG-based code analysis, developed as part of this code analysis project.

## Overview

This RAG pipeline:
1. **Ingests** Java source code, tests, and documentation from the repository
2. **Chunks** documents using a file-aware strategy optimized for code
3. **Embeds** chunks using sentence-transformers (local, no API key needed)
4. **Retrieves** relevant context using cosine similarity search
5. **Generates** answers using an LLM with grounded prompts
6. **Analyzes** architecture issues and suggests improvements

## Project Structure

```
rag/
├── main.py                    # CLI entry point
├── config.py                  # Configuration settings  
├── ingest.py                  # Document ingestion, extraction, chunking
├── vector_store.py            # Embedding and vector search
├── langchain_integration.py   # LangChain compatibility (optional)
├── architecture_agent.py      # Architecture analysis facade
├── requirements.txt           # Python dependencies
├── pipeline/                  # RAG pipeline implementations
│   ├── __init__.py           # Base classes and interfaces
│   ├── native_pipeline.py    # Native RAG with OpenAI/Ollama
│   └── langchain_pipeline.py # LangChain-based RAG
├── analysis/                  # Architecture analysis modules
│   ├── models.py             # Data classes (DependencyInfo, etc.)
│   ├── dependency_analyzer.py # Static dependency analysis
│   ├── issue_detector.py     # Architecture issue detection
│   └── refactoring_advisor.py # Refactoring recommendations
├── utils/                     # Shared utilities
│   ├── lazy_imports.py       # Lazy loading utilities
│   ├── logging_config.py     # Consistent logging setup
│   └── document_formatter.py # Document formatting for RAG
├── tests/                     # Test suite
│   └── test_rag_queries.py   # RAG query tests
└── index/                     # Saved embeddings and documents
```

## TL;DR - Quick Start

```bash
# 1. Setup (first-time)
cd rag
pip install -r requirements.txt
ollama serve &              # Start local LLM (in background)
ollama pull tinyllama       # Download model (default, fast)
python main.py --build      # Build vector index

# 2. Query the codebase
python main.py --query "How does StringSubstitutor work?"

# 3. Run architecture analysis
python main.py --analyze
```

## Design Decisions

This section documents the key architectural choices made in building this RAG system.

### Chunking Strategy

We use a **file-level chunking strategy with class/method awareness**:

| Strategy | Why We Chose It |
|----------|-----------------|
| File-level primary | Java files = natural semantic units (one class per file) |
| Preserve imports | Essential for understanding dependencies |
| Keep Javadocs | Documentation stays with code |
| Split large files | Maintain embedding model limits while preserving context |

For files under 8000 chars (~2000 tokens): kept as single chunk
For larger files: split at method boundaries with class header preserved

### Retrieval Strategy

We use **hybrid search** combining semantic and lexical matching:

| Component | Approach | Rationale |
|-----------|----------|-----------|
| Base retrieval | Cosine similarity on embeddings | Captures semantic meaning |
| Keyword boosting | Exact class/file name matches | Important for code (CamelCase names) |
| Source prioritization | `src/main/` weighted higher than changelogs | Prefer actual implementation |
| Folder detection | Boosts `package-info.java` for folder queries | Package docs are most informative |

Top-k (default 5) retrieval with diversity: for folder queries, we limit chunks from the same file to ensure coverage across the package.

### Prompting Strategy

The prompts are designed to **ground answers in retrieved context** and minimize hallucination:

| Technique | Implementation |
|-----------|----------------|
| Context-first prompting | Retrieved code snippets placed before the question |
| Query type detection | Different prompts for class queries vs. folder overview |
| Source attribution | LLM instructed to reference specific files |
| Uncertainty handling | Explicit instruction to say "I don't know" if context insufficient |
| Post-hoc verification | Check if mentioned classes exist in retrieved sources |

**Prompt structure:**
```
System: You are a Java expert answering about Apache Commons Text.
        Base your answer ONLY on the provided code snippets.
        If the context doesn't contain the answer, say so.
        
Context: [Retrieved code chunks with file paths]

Question: {user_query}
```

### Dependency Analysis Approach

Static analysis using **AST-free import parsing** with **percentile-based thresholds**:

| Aspect | Approach | Rationale |
|--------|----------|-----------|
| Parsing | Regex on import statements | Fast, sufficient for Java; no external parser needed |
| Graph building | Build adjacency list of class→dependencies | Enables cycle detection, coupling metrics |
| Issue detection | Percentile-based thresholds | Adapts to codebase size (see [Percentile-Based Thresholds](#percentile-based-thresholds)) |
| Recommendations | Severity-sorted with concrete examples | Critical issues first, actionable suggestions |

We chose regex over full AST parsing because:
1. Java imports are line-based and predictable
2. No external dependencies (no need for javalang, ANTLR)
3. Fast enough for interactive use

## Installation

### Requirements

- **Python 3.9+** (uses modern type hints)
- **Ollama** (recommended) or **OpenAI API key** for LLM inference
- ~2GB disk space for embeddings model (downloaded on first run)

### Setup

```bash
cd rag
pip install -r requirements.txt
```

## Usage

### 1. Build the Index (First Time)

```bash
python main.py --build
```

This creates embeddings for all source files and saves them to `rag/index/`.

### 2. Ask Questions

**Single query:**
```bash
python main.py --query "How does StringSubstitutor handle variable substitution?"
```

**Interactive mode:**
```bash
python main.py --interactive
```

### 3. Architecture Analysis

Run architecture analysis to detect issues and get recommendations:

```bash
python main.py --analyze
```

This will:
- Analyze dependencies using static analysis (imports, package references)
- Detect architecture issues (dependency magnets, cycles, oversized modules)
- Generate concrete refactoring recommendations with rationale

### 4. Programmatic Usage

```python
from main import create_rag_pipeline
from architecture_agent import analyze_architecture

# RAG Q&A
pipeline = create_rag_pipeline()
result = pipeline.query("What text similarity algorithms are available?")
print(result.answer)  # Note: result is now a RAGResult dataclass
print(result.sources)

# Architecture analysis
analysis = analyze_architecture()
for issue in analysis["issues"]:
    print(f"[{issue.severity}] {issue.title}")
```

### 5. Running Tests

```bash
cd rag
python -m pytest tests/ -v
```

The test suite includes:
- Query tests for specific classes (e.g., StringSubstitutor)
- Folder/package query tests
- General repository questions
- Tests for queries that should fail (irretrievable info)

## Configuration

### LLM Backend

The pipeline supports both **local LLMs (Ollama)** and **OpenAI**.

#### Option 1: Local LLM with Ollama (Recommended)

No API key required! Uses Ollama for local inference.

**1. Install Ollama:**
```bash
# macOS
brew install ollama

# Or download from https://ollama.ai
```

**2. Start the Ollama server:**
```bash
ollama serve
```

**3. Pull a model:**
```bash
ollama pull llama3.2. # or use tinyllama
```

**4. Run queries:**
```bash
python main.py --query "How does StringSubstitutor work?"
```

**Recommended models:**

| Model | Size | Description |
|-------|------|-------------|
| `tinyllama` | 600MB | Very fast, basic quality (default) |
| `llama3.2` | 2GB | Good balance of speed and quality |
| `codellama` | 4GB | Optimized for code understanding |
| `mistral` | 4GB | Fast and capable |
| `phi3` | 2GB | Smaller, runs on limited hardware |

**Note:** `tinyllama` is the default for fast responses, but larger models like `llama3.2` or `codellama` give significantly better answers, especially for package overview queries.

**Change the model:**
```bash
export OLLAMA_MODEL=llama3.2
python main.py --query "..."
```

#### Option 2: OpenAI

Set your API key to use OpenAI instead:
```bash
export OPENAI_API_KEY="your-key-here"
```

When `OPENAI_API_KEY` is set, the pipeline automatically uses OpenAI.

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OLLAMA_MODEL` | Ollama model to use | `tinyllama` |
| `OLLAMA_URL` | Ollama API URL | `http://localhost:11434/v1` |
| `OPENAI_API_KEY` | OpenAI API key (uses OpenAI if set) | (none) |
| `OPENAI_MODEL` | OpenAI model to use | `gpt-4o-mini` |

## Understanding Relevance Scores

The sources shown after each answer include relevance scores and line numbers:

```
- src/main/java/.../StringSubstitutor.java (L45-120) [relevance: 0.65]
- src/test/java/.../StringSubstitutorTest.java (L1-50) [relevance: 0.52]
```

### How to interpret scores:

| Score | Meaning |
|-------|---------|
| > 0.6 | Highly relevant - confident match |
| 0.4 - 0.6 | Moderately relevant - likely useful |
| 0.3 - 0.4 | Low relevance - may not be helpful |
| < 0.3 | Very low - probably not what you need |

**Note:** Scores are cosine similarity enhanced with keyword matching. Scores above 1.0 are possible due to additive boosts.

### Hybrid Search

The search uses a combination of:
1. **Semantic similarity** - embedding-based cosine similarity
2. **Keyword matching** - boosts for exact class/file name matches
3. **Source prioritization** - prefers actual code over changelogs

**Automatic boosts applied:**

| Match Type | Boost |
|------------|-------|
| Exact class file match (e.g., "StringSubstitutor" → `StringSubstitutor.java`) | +0.5 |
| Folder/package match (e.g., "lookup folder" → files in `/lookup/`) | +0.4 |
| `package-info.java` for folder queries | +0.8 |
| Main source code (`src/main/java`) | ×1.3 |
| Test files | ×1.1 |
| Changelogs and release notes | ×0.5 (penalty) |

## Query Types

The RAG pipeline automatically detects and optimizes for different query types:

### Class/Code Queries (Default)

Ask about specific classes, methods, or functionality:

```bash
python main.py --query "How does StringSubstitutor work?"
python main.py --query "What methods does TextStringBuilder have?"
```

The search prioritizes exact file matches and provides detailed code context.

### Folder/Package Queries

Ask about entire packages or folders to get an overview:

```bash
python main.py --query "What functionality is in the lookup folder?"
python main.py --query "What does the translate package contain?"
python main.py --query "Show me what's in the similarity directory"
```

Keywords like "folder", "package", "directory", or "module" trigger package-overview mode:
- Prioritizes `package-info.java` (package documentation)
- Returns diverse files from the folder (not just chunks of one file)
- LLM prompt asks for package overview, not class deep-dive

## Hallucination Detection

The pipeline includes basic hallucination detection to warn when the LLM may have invented details:

```
⚠️ **Warning: Possible inaccuracies detected**
The following references could not be verified in the source code:
  - Class 'FakeClass' not found in codebase
```

This checks whether class names mentioned in the answer actually exist in the retrieved sources.

## LangChain Integration (Optional)

The pipeline includes optional LangChain integration for advanced features like chains, memory, and agents.

### Installation

```bash
pip install langchain langchain-core langchain-community langchain-ollama
```

Or uncomment the LangChain lines in `requirements.txt`.

### Using the Custom Retriever

The `HybridCodeRetriever` wraps our hybrid search logic in a LangChain-compatible interface:

```python
from vector_store import SimpleVectorStore
from langchain_integration import get_langchain_retriever

# Load vector store
vs = SimpleVectorStore()
vs.load_index()

# Create LangChain retriever (preserves all custom boosting logic)
retriever = get_langchain_retriever(vs, top_k=5)

# Use with any LangChain chain
docs = retriever.invoke("How does StringSubstitutor work?")
for doc in docs:
    print(f"{doc.metadata['path']}: {doc.metadata['relevance_score']:.2f}")
```

### Using RAG Chains

Create a complete RAG chain with one function:

```python
from vector_store import SimpleVectorStore
from langchain_integration import create_rag_chain

vs = SimpleVectorStore()
vs.load_index()

# Create chain (uses Ollama by default)
chain = create_rag_chain(vs, model="llama3.2")

# Query
answer = chain.invoke("What string similarity algorithms are available?")
print(answer)
```

### Conversational Memory

For interactive sessions with follow-up questions:

```python
from langchain_integration import create_conversational_chain

vs = SimpleVectorStore()
vs.load_index()

chain, memory = create_conversational_chain(vs, model="llama3.2")

# First question
result1 = chain.invoke(
    {"input": "What is StringSubstitutor?"},
    config={"configurable": {"session_id": "user1"}}
)

# Follow-up (remembers context)
result2 = chain.invoke(
    {"input": "How do I use variable prefixes with it?"},
    config={"configurable": {"session_id": "user1"}}
)
```

### Quick One-Shot Query

For simple usage:

```python
from langchain_integration import quick_query

answer = quick_query("What escape utilities are available?", model="tinyllama")
```

### Why Use LangChain?

| Feature | Without LangChain | With LangChain |
|---------|-------------------|----------------|
| Basic RAG | ✅ Built-in | ✅ `create_rag_chain()` |
| Conversation memory | ❌ | ✅ `create_conversational_chain()` |
| Agent tools | ❌ | ✅ Use retriever with agents |
| Streaming responses | ❌ | ✅ Built-in |
| Model switching | Manual | Automatic via LangChain |

The custom retriever preserves all our hybrid search features (keyword boosting, folder detection, source prioritization) while making the pipeline compatible with the LangChain ecosystem.

## Safe Mode

For systems with limited RAM, use safe mode:
```bash
python main.py --build --safe
```

This uses:
- Sequential processing (1 worker)
- Small batch sizes for embeddings
- Lower memory footprint

## Architecture Analysis

The architecture agent performs static analysis and issue detection using **percentile-based thresholds** that adapt to your codebase's distribution.

### Static Analysis
- Extracts imports and package dependencies
- Builds dependency graph between classes
- Computes metrics (LOC, method count, coupling)
- Calculates distribution statistics (min, median, mean, max)

### Percentile-Based Thresholds

Instead of hardcoded thresholds, issues are flagged based on where classes fall in the distribution:

| Metric | Percentile | Meaning |
|--------|------------|---------|
| Lines of Code | 90th | Top 10% largest classes flagged |
| Method Count | 90th | Top 10% by number of methods |
| Dependencies | 85th | Top 15% by outgoing dependencies |
| Dependents | 85th | Top 15% by incoming dependencies |

**Why percentile-based?** A class with 500 LOC might be "oversized" in a microservices codebase but perfectly normal in a utility library. Percentile thresholds automatically adapt.

To adjust sensitivity, edit [analysis/issue_detector.py](analysis/issue_detector.py):
```python
LOC_PERCENTILE = 90        # Flag top 10% (lower = more strict)
METHOD_PERCENTILE = 90     # Flag top 10%
DEPENDENCY_PERCENTILE = 85 # Flag top 15%
DEPENDENT_PERCENTILE = 85  # Flag top 15%
```

### Viewing Distribution Statistics

When you run analysis, you'll see codebase statistics:

```
CODEBASE DISTRIBUTION STATISTICS:
  loc: min=22, median=110, mean=284.9, max=3200
    → threshold (p90): 711.7
  methods: min=0, median=2, mean=9.7, max=179
    → threshold (p90): 20.7
  dependencies: min=0, median=2, mean=3.4, max=17
    → threshold (p85): 6.5
```

This helps you understand what "normal" looks like in your codebase and why certain classes were flagged.

### Issue Detection

| Issue Type | Description |
|------------|-------------|
| Dependency Magnet | Classes imported by many others (top percentile), hard to change |
| Cyclic Dependency | Circular dependencies between classes |
| Oversized Module | Classes with too many lines/methods (top percentile) |
| God Class | Classes with too many dependencies (top percentile) |
| Unclear Separation | High coupling between packages |

### Evidence in Issues

Each issue includes detailed evidence with both the value and codebase context:

```python
evidence = {
    "lines_of_code": 1582,
    "method_count": 56,
    "loc_percentile_threshold": 711.7,
    "codebase_median_loc": 110,
    "codebase_max_loc": 3200
}
```

### Recommendations

Recommendations are organized by priority:

#### 🔴 CRITICAL (High Severity)
These appear first and should be addressed to resolve major architecture issues:
- **Cyclic dependencies** - Break circular imports between classes
- **Decompose oversized classes** - Split large classes (top 10% by LOC/methods)
- **Reduce high coupling** - Address classes with too many dependencies

#### ✅ GOOD PRACTICES (Medium/Low Severity)  
After critical issues, you'll see a separator:
```
✅ Critical issues addressed above.
   The following are GOOD PRACTICES for further improvement:
```

These are optional improvements for better code quality:
- **Extract interfaces** for dependency magnets
- **Clarify package boundaries** for cross-package coupling

#### Recommendation Details

Each recommendation includes:
- **Description**: What to do
- **Rationale**: Why it helps (grounded in evidence)
- **Quality Impact**: Effect on maintainability, testability, evolvability
- **Effort**: Low/Medium/High
- **Concrete Examples**: Specific files and suggested names

## Output Format

Every answer includes:
- **Answer**: LLM-generated response grounded in retrieved context
- **Sources**: List of files used with relevance scores
- **Uncertainty flag**: Warns when retrieved context has low relevance

## Uncertainty Handling

The pipeline explicitly handles uncertainty:
- If retrieval scores are below 0.3, it returns a clear "insufficient information" message
- LLM is instructed to admit when context doesn't contain the answer
- Sources and relevance scores are always shown for transparency

## Example Queries

### Class/Code Questions
- "How do I use StringSubstitutor to replace variables in a string?"
- "What string similarity metrics are available?"
- "How does the diff algorithm work?"
- "What escape utilities does StringEscapeUtils provide?"
- "How do I generate random strings?"

### Package/Folder Overview
- "What functionality is in the lookup folder?"
- "What does the translate package do?"
- "Show me the similarity classes"
- "What's in the matcher package?"

## Troubleshooting

### Low relevance scores
If you're getting low relevance scores (< 0.3), try:
1. Use exact class names (CamelCase): "StringSubstitutor" not "string substitutor"
2. For packages, use keywords like "folder" or "package": "lookup folder"
3. Be specific: "StringSubstitutor variable replacement" instead of "how to replace text"

### Ollama connection errors
```
Error connecting to Ollama
```
Make sure Ollama is running:
```bash
ollama serve
```

### Out of memory during build
Use safe mode:
```bash
python main.py --build --safe
```

### Hallucination warnings
If you see "Possible inaccuracies detected", the LLM mentioned classes or concepts not found in the retrieved sources. Verify the answer against the listed source files.

---

## AI Usage Statement

This project was developed with the assistance of AI tools:

- **ChatGPT**: Used for initial drafting and organizing the steps to solve this task
- **GitHub Copilot**: Used for coding assistance including:
  - Initial scaffolding and basic structure
  - Refactoring and debugging help
  - LangChain integration
  - Type hinting coverage, method signatures, code commenting, and README creation
- **Microsoft Copilot**: Used for grammar/spell checking and writing suggestions in the final report
