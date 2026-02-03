"""
Lazy Import Utilities

Provides consistent lazy loading patterns to avoid importing heavy dependencies
until they are actually needed. This improves startup time and reduces memory
usage when only a subset of features is used.
"""

from typing import Any, Callable, Optional, TypeVar, Generic
import importlib
import sys

T = TypeVar('T')


class LazyModule(Generic[T]):
    """
    A lazy module loader that defers importing until first access.
    
    Usage:
        numpy = LazyModule('numpy')
        # numpy is not imported yet
        
        arr = numpy.array([1, 2, 3])  # Now numpy is imported
    """
    
    def __init__(self, module_name: str) -> None:
        self._module_name = module_name
        self._module: Optional[Any] = None
    
    def _load(self) -> Any:
        if self._module is None:
            self._module = importlib.import_module(self._module_name)
        return self._module
    
    def __getattr__(self, name: str) -> Any:
        return getattr(self._load(), name)
    
    def __repr__(self) -> str:
        if self._module is None:
            return f"<LazyModule '{self._module_name}' (not loaded)>"
        return f"<LazyModule '{self._module_name}' (loaded)>"


def lazy_import(module_name: str) -> LazyModule[Any]:
    """
    Create a lazy module reference.
    
    Args:
        module_name: Fully qualified module name (e.g., 'numpy', 'torch.cuda')
        
    Returns:
        LazyModule that loads on first attribute access
        
    Example:
        np = lazy_import('numpy')
        torch = lazy_import('torch')
        
        # Modules not loaded yet
        arr = np.array([1, 2, 3])  # numpy loaded here
    """
    return LazyModule(module_name)


def check_import_available(module_name: str) -> bool:
    """
    Check if a module can be imported without actually importing it.
    
    Args:
        module_name: Module name to check
        
    Returns:
        True if the module is available, False otherwise
    """
    if module_name in sys.modules:
        return True
    
    spec = importlib.util.find_spec(module_name)
    return spec is not None


class LazyClassLoader:
    """
    Lazily loads a class from a module.
    
    Usage:
        SentenceTransformer = LazyClassLoader(
            'sentence_transformers', 
            'SentenceTransformer'
        )
        model = SentenceTransformer()  # Module loaded here
    """
    
    def __init__(self, module_name: str, class_name: str) -> None:
        self._module_name = module_name
        self._class_name = class_name
        self._class: Optional[type] = None
    
    def _load(self) -> type:
        if self._class is None:
            module = importlib.import_module(self._module_name)
            self._class = getattr(module, self._class_name)
        return self._class
    
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._load()(*args, **kwargs)
    
    def __repr__(self) -> str:
        return f"<LazyClassLoader '{self._module_name}.{self._class_name}'>"


# Pre-configured lazy imports for common heavy dependencies
def get_numpy() -> Any:
    """Get numpy module (lazy loaded)."""
    return lazy_import('numpy')._load()


def get_sentence_transformer_class() -> type:
    """Get SentenceTransformer class (lazy loaded)."""
    loader = LazyClassLoader('sentence_transformers', 'SentenceTransformer')
    return loader._load()


def get_openai_client_class() -> type:
    """Get OpenAI client class (lazy loaded)."""
    loader = LazyClassLoader('openai', 'OpenAI')
    return loader._load()
