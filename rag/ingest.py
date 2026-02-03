"""
Document ingestion, extraction, and chunking for the RAG pipeline.

CHUNKING STRATEGY RATIONALE:
----------------------------
We use a **file-level chunking strategy with class/method awareness** for the following reasons:

1. **Semantic Coherence**: In Java codebases, a single file typically contains one public class
   with related methods. This provides natural semantic boundaries that preserve context.

2. **Import/Package Context**: File-level chunks preserve import statements and package 
   declarations which are essential for understanding class dependencies and usage.

3. **Javadoc Preservation**: Method and class documentation stays together with the code,
   enabling the LLM to provide accurate, documented answers.

4. **Manageable Size**: Most Java files in well-structured projects like Apache Commons
   are reasonably sized (100-500 lines), fitting well within embedding model limits.

5. **For very large files**: We apply secondary chunking by extracting individual methods
   with their Javadocs to ensure we don't exceed token limits while maintaining context.
"""

import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from config import REPO_ROOT, MAX_CHUNK_CHARS, OVERLAP_CHARS
from utils.logging_config import get_logger, log_progress

logger = get_logger(__name__)


def read_text(path: Path) -> str:
    """
    Read file content with error handling.
    
    Args:
        path: Path to the file
        
    Returns:
        File content as string
    """
    return path.read_text(encoding="utf-8", errors="replace")


def extract_java_class_info(content: str) -> dict[str, str]:
    """
    Extract class name and package from Java source.
    
    Args:
        content: Java source code
        
    Returns:
        Dict with 'package' and 'class_name' keys
    """
    package_match = re.search(r'package\s+([\w.]+);', content)
    
    class_match = re.search(
        r'^(?:public\s+)?(?:abstract\s+)?(?:final\s+)?(?:class|interface|enum)\s+([A-Z]\w*)',
        content,
        re.MULTILINE
    )
    
    return {
        "package": package_match.group(1) if package_match else "",
        "class_name": class_match.group(1) if class_match else ""
    }


def get_line_number(content: str, char_pos: int) -> int:
    """
    Get the 1-based line number for a character position.
    
    Args:
        content: Full text content
        char_pos: Character position
        
    Returns:
        1-based line number
    """
    return content[:char_pos].count('\n') + 1


def extract_javadoc(content: str) -> str:
    """Extract the main class-level Javadoc comment."""
    match = re.search(r'/\*\*[\s\S]*?\*/\s*(?=public|class|interface|enum|abstract)', content)
    return match.group(0) if match else ""


def extract_imports(content: str) -> list[str]:
    """Extract all import statements from Java source."""
    return re.findall(r'import\s+([\w.]+(?:\.\*)?);', content)


def chunk_java_file(path: str, content: str) -> list[dict[str, Any]]:
    """
    Chunk a Java file intelligently.
    
    For smaller files: keep as single chunk (preserves full context)
    For larger files: split by methods while keeping class header
    
    Args:
        path: Relative file path
        content: File content
        
    Returns:
        List of document chunks
    """
    chunks: list[dict[str, Any]] = []
    class_info = extract_java_class_info(content)
    
    if len(content) <= MAX_CHUNK_CHARS:
        # Small file - keep as single chunk
        total_lines = content.count('\n') + 1
        chunks.append({
            "id": f"{path}::full",
            "path": path,
            "type": "java",
            "class_name": class_info["class_name"],
            "package": class_info["package"],
            "chunk_type": "full_file",
            "start_line": 1,
            "end_line": total_lines,
            "text": content
        })
    else:
        # Large file - chunk with overlap
        header_match = re.search(
            r'^(.*?(?:public|abstract|final)\s+(?:class|interface|enum)\s+\w+[^{]*\{)',
            content, re.DOTALL
        )
        header = header_match.group(1) if header_match else content[:500]
        
        current_pos = len(header)
        chunk_num = 0
        max_chunks = 100  # Safety limit
        
        while current_pos < len(content) and chunk_num < max_chunks:
            effective_chunk_size = max(500, MAX_CHUNK_CHARS - len(header))
            chunk_end = min(current_pos + effective_chunk_size, len(content))
            
            # Try to end at a method boundary
            if chunk_end < len(content):
                boundary_search = content[max(0, chunk_end - 200):min(len(content), chunk_end + 200)]
                brace_match = re.search(r'\n\s*\}\s*\n', boundary_search)
                if brace_match:
                    chunk_end = max(0, chunk_end - 200) + brace_match.end()
            
            chunk_text = header + "\n// ... (continued from file) ...\n\n" + content[current_pos:chunk_end]
            
            start_line = get_line_number(content, current_pos)
            end_line = get_line_number(content, chunk_end)
            
            chunks.append({
                "id": f"{path}::chunk_{chunk_num}",
                "path": path,
                "type": "java",
                "class_name": class_info["class_name"],
                "package": class_info["package"],
                "chunk_type": "partial",
                "start_line": start_line,
                "end_line": end_line,
                "text": chunk_text
            })
            
            overlap = min(OVERLAP_CHARS, effective_chunk_size // 2)
            new_pos = chunk_end - overlap
            if new_pos <= current_pos:
                new_pos = current_pos + max(1, effective_chunk_size // 2)
            current_pos = new_pos
            chunk_num += 1
        
        if chunk_num >= max_chunks:
            logger.warning(f"Stopped chunking at {max_chunks} chunks for {path}")
    
    return chunks


def chunk_document(path: str, doc_type: str, content: str) -> list[dict[str, Any]]:
    """
    Chunk a document based on its type.
    
    Args:
        path: Relative file path
        doc_type: Document type ('java', 'java_test', etc.)
        content: File content
        
    Returns:
        List of document chunks
    """
    if doc_type in ("java", "java_test"):
        return chunk_java_file(path, content)
    
    # For non-Java files - simple chunking
    chunks: list[dict[str, Any]] = []
    
    if len(content) <= MAX_CHUNK_CHARS:
        total_lines = content.count('\n') + 1
        chunks.append({
            "id": f"{path}::full",
            "path": path,
            "type": doc_type,
            "chunk_type": "full_file",
            "start_line": 1,
            "end_line": total_lines,
            "text": content
        })
    else:
        step = MAX_CHUNK_CHARS - OVERLAP_CHARS
        for i in range(0, len(content), step):
            chunk_text = content[i:i + MAX_CHUNK_CHARS]
            start_line = get_line_number(content, i)
            end_line = get_line_number(content, min(i + MAX_CHUNK_CHARS, len(content)))
            chunks.append({
                "id": f"{path}::chunk_{i // step}",
                "path": path,
                "type": doc_type,
                "chunk_type": "partial",
                "start_line": start_line,
                "end_line": end_line,
                "text": chunk_text
            })
    
    return chunks


def _process_file(args: tuple[Path, Path, str]) -> list[dict[str, Any]]:
    """Process a single file (for parallel execution)."""
    path, repo_root, doc_type = args
    content = read_text(path)
    rel_path = str(path.relative_to(repo_root))
    return chunk_document(rel_path, doc_type, content)


def ingest(
    repo_root: Path = REPO_ROOT,
    parallel: bool = True,
    max_workers: int = 4,
    max_java_files: int = -1,
    max_test_files: int = -1
) -> list[dict[str, Any]]:
    """
    Ingest all relevant documents from the repository.
    
    Args:
        repo_root: Path to repository root
        parallel: Use parallel processing for faster ingestion
        max_workers: Number of parallel workers
        max_java_files: Maximum Java source files to process (-1 for all)
        max_test_files: Maximum test files to process (-1 for all)
        
    Returns:
        List of document dictionaries with metadata
    """
    docs: list[dict[str, Any]] = []

    # Java sources (main code)
    java_files = list(repo_root.glob("src/main/java/**/*.java"))
    total_java = len(java_files)
    if max_java_files >= 0 and len(java_files) > max_java_files:
        java_files = java_files[:max_java_files]
        logger.info(f"Found {total_java} Java source files (processing {len(java_files)})")
    else:
        logger.info(f"Found {total_java} Java source files")

    # Java test sources
    test_files = list(repo_root.glob("src/test/java/**/*.java"))
    total_test = len(test_files)
    if max_test_files >= 0 and len(test_files) > max_test_files:
        test_files = test_files[:max_test_files]
        logger.info(f"Found {total_test} Java test files (processing {len(test_files)})")
    else:
        logger.info(f"Found {total_test} Java test files")

    total_files = len(java_files) + len(test_files)

    if parallel and total_files > 10:
        docs.extend(_process_parallel(java_files, test_files, repo_root, max_workers))
    else:
        docs.extend(_process_sequential(java_files, test_files, repo_root, total_files))

    # Additional files
    docs.extend(_process_additional_files(repo_root))

    logger.info(f"Total: {len(docs)} chunks ready for embedding")
    return docs


def _process_parallel(
    java_files: list[Path],
    test_files: list[Path],
    repo_root: Path,
    max_workers: int
) -> list[dict[str, Any]]:
    """Process files in parallel."""
    docs: list[dict[str, Any]] = []
    
    all_tasks = (
        [(p, repo_root, "java") for p in java_files] +
        [(p, repo_root, "java_test") for p in test_files]
    )
    
    processed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_process_file, task) for task in all_tasks]
        for future in as_completed(futures):
            docs.extend(future.result())
            processed += 1
            log_progress(logger, processed, len(all_tasks), "Processing", interval=50)
    
    logger.info(f"Processed {len(all_tasks)} files → {len(docs)} chunks")
    return docs


def _process_sequential(
    java_files: list[Path],
    test_files: list[Path],
    repo_root: Path,
    total_files: int
) -> list[dict[str, Any]]:
    """Process files sequentially."""
    docs: list[dict[str, Any]] = []
    processed = 0
    
    logger.info(f"Processing {total_files} files...")
    
    for p in java_files:
        try:
            chunks = _process_file((p, repo_root, "java"))
            docs.extend(chunks)
        except Exception as e:
            logger.error(f"Error processing {p.name}: {e}")
        processed += 1
        log_progress(logger, processed, total_files, "Processing", interval=25)
    
    for p in test_files:
        try:
            chunks = _process_file((p, repo_root, "java_test"))
            docs.extend(chunks)
        except Exception as e:
            logger.error(f"Error processing {p.name}: {e}")
        processed += 1
        log_progress(logger, processed, total_files, "Processing", interval=25)
    
    logger.info(f"Processed {total_files} files → {len(docs)} chunks")
    return docs


def _process_additional_files(repo_root: Path) -> list[dict[str, Any]]:
    """Process additional documentation files."""
    docs: list[dict[str, Any]] = []
    
    # Build file
    pom = repo_root / "pom.xml"
    if pom.exists():
        docs.extend(chunk_document("pom.xml", "pom", read_text(pom)))

    # Documentation files
    for name in ["README.md", "README.adoc", "README.txt", "RELEASE-NOTES.txt"]:
        rp = repo_root / name
        if rp.exists():
            docs.extend(chunk_document(name, "documentation", read_text(rp)))

    # Changes/release notes
    changes = repo_root / "src" / "changes" / "changes.xml"
    if changes.exists():
        docs.extend(chunk_document("src/changes/changes.xml", "changelog", read_text(changes)))

    return docs
