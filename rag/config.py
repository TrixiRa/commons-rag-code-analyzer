"""
Configuration settings for the RAG pipeline.
"""

from pathlib import Path
from typing import Literal


# ============================================================================
# PATHS
# ============================================================================

REPO_ROOT: Path = Path(__file__).parent.parent.resolve()
INDEX_DIR: Path = Path(__file__).parent / "index"


# ============================================================================
# CHUNKING CONFIGURATION
# ============================================================================

MAX_CHUNK_CHARS: int = 8000  # ~2000 tokens, safe for most embedding models
OVERLAP_CHARS: int = 200     # Overlap between chunks for continuity


# ============================================================================
# EMBEDDING MODEL OPTIONS
# ============================================================================

# Faster -> Slower, Less accurate -> More accurate
EMBEDDING_MODELS: dict[str, str] = {
    "fast": "all-MiniLM-L6-v2",      # ~23M params, fastest, good quality
    "balanced": "all-MiniLM-L12-v2", # ~33M params, balanced
    "accurate": "all-mpnet-base-v2", # ~110M params, best quality, slower
}
DEFAULT_MODEL: str = "fast"  # Change to "balanced" or "accurate" if needed


# ============================================================================
# SEARCH CONFIGURATION
# ============================================================================

# Relevance score thresholds
LOW_RELEVANCE_THRESHOLD: float = 0.3

# Boosting factors for hybrid search
PACKAGE_INFO_BOOST: float = 0.8
FOLDER_MATCH_BOOST: float = 0.4
PARTIAL_FOLDER_BOOST: float = 0.2
EXACT_FILE_BOOST: float = 0.5
CLASS_IN_PATH_BOOST: float = 1.5
TERM_MATCH_BOOST: float = 0.1
MAIN_SOURCE_BOOST: float = 1.3
TEST_SOURCE_BOOST: float = 1.1
CHANGELOG_PENALTY: float = 0.5


# ============================================================================
# LLM CONFIGURATION
# ============================================================================

DEFAULT_LLM_PROVIDER: Literal["ollama", "openai"] = "ollama"
DEFAULT_OLLAMA_MODEL: str = "tinyllama"
DEFAULT_OLLAMA_URL: str = "http://localhost:11434/v1"


# ============================================================================
# DEVICE DETECTION
# ============================================================================

DeviceType = Literal["cuda", "mps", "cpu"]


def detect_device() -> DeviceType:
    """
    Detect the best available device for embedding generation.
    
    Returns:
        'cuda' for NVIDIA GPU, 'mps' for Apple Silicon, or 'cpu'
    """
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"
