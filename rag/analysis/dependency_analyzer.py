"""
Dependency Analyzer

Performs static analysis on Java source files to extract
dependency information and build dependency graphs.
"""

import re
from pathlib import Path
from collections import defaultdict
from typing import Optional

from analysis.models import DependencyInfo
from utils.logging_config import get_logger

logger = get_logger(__name__)


def read_text(path: Path) -> str:
    """Read file content with error handling."""
    return path.read_text(encoding="utf-8", errors="replace")


def extract_java_class_info(content: str) -> dict[str, str]:
    """Extract class name and package from Java source."""
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


class DependencyAnalyzer:
    """Analyzes dependencies in a Java codebase using static analysis."""
    
    # Default package prefix for Apache Commons Text
    DEFAULT_PACKAGE_PREFIX = "org.apache.commons.text"
    
    def __init__(
        self, 
        repo_root: Path,
        package_prefix: Optional[str] = None
    ) -> None:
        """
        Initialize the dependency analyzer.
        
        Args:
            repo_root: Path to repository root
            package_prefix: Package prefix for internal dependencies
        """
        self.repo_root = repo_root
        self.classes: dict[str, DependencyInfo] = {}
        self.package_prefix = package_prefix or self.DEFAULT_PACKAGE_PREFIX
    
    def analyze(self) -> dict[str, DependencyInfo]:
        """
        Analyze all Java source files and extract dependency information.
        
        Returns:
            Dictionary mapping fully qualified class names to DependencyInfo
        """
        logger.info("Analyzing dependencies...")
        
        java_files = list(self.repo_root.glob("src/main/java/**/*.java"))
        logger.info(f"Found {len(java_files)} Java source files")
        
        for path in java_files:
            self._analyze_file(path)
        
        self._classify_dependencies()
        
        logger.info(f"Analyzed {len(self.classes)} classes")
        return self.classes
    
    def _analyze_file(self, path: Path) -> None:
        """Analyze a single Java file."""
        try:
            content = read_text(path)
            rel_path = str(path.relative_to(self.repo_root))
            
            class_info = extract_java_class_info(content)
            if not class_info["class_name"]:
                return
            
            # Extract imports with line numbers
            imports: list[str] = []
            import_lines: dict[str, int] = {}
            for i, line in enumerate(content.split('\n'), 1):
                match = re.match(r'^\s*import\s+([\w.]+(?:\.\*)?);', line)
                if match:
                    imp = match.group(1)
                    imports.append(imp)
                    import_lines[imp] = i
            
            # Count methods
            method_pattern = r'(?:public|private|protected)\s+(?:static\s+)?(?:final\s+)?[\w<>\[\],\s]+\s+\w+\s*\([^)]*\)\s*(?:throws\s+[\w,\s]+)?\s*\{'
            methods = re.findall(method_pattern, content)
            
            # Count lines
            all_lines = content.split('\n')
            total_lines = len(all_lines)
            nonblank_lines = [
                l for l in all_lines 
                if l.strip() and not l.strip().startswith('//') and not l.strip().startswith('*')
            ]
            
            fqn = f"{class_info['package']}.{class_info['class_name']}"
            self.classes[fqn] = DependencyInfo(
                path=rel_path,
                package=class_info["package"],
                class_name=class_info["class_name"],
                imports=imports,
                import_lines=import_lines,
                lines_of_code=total_lines,
                lines_of_code_nonblank=len(nonblank_lines),
                method_count=len(methods)
            )
        except Exception as e:
            logger.warning(f"Could not analyze {path}: {e}")
    
    def _classify_dependencies(self) -> None:
        """Classify imports as internal (project) or external dependencies."""
        project_packages = {info.package for info in self.classes.values()}
        
        for fqn, info in self.classes.items():
            for imp in info.imports:
                imp_package = '.'.join(imp.split('.')[:-1])
                if imp.startswith(self.package_prefix) or imp_package in project_packages:
                    info.internal_deps.append(imp)
                else:
                    info.external_deps.append(imp)
    
    def get_dependency_graph(self) -> dict[str, set[str]]:
        """Build a dependency graph (class -> classes it depends on)."""
        graph: dict[str, set[str]] = defaultdict(set)
        
        name_to_fqn: dict[str, str] = {}
        for fqn, info in self.classes.items():
            name_to_fqn[info.class_name] = fqn
        
        for fqn, info in self.classes.items():
            for dep in info.internal_deps:
                dep_class = dep.split('.')[-1]
                if dep_class in name_to_fqn:
                    graph[fqn].add(name_to_fqn[dep_class])
                elif not dep.endswith('*'):
                    graph[fqn].add(dep)
        
        return dict(graph)
    
    def get_reverse_dependency_graph(self) -> dict[str, set[str]]:
        """Build reverse dependency graph (class -> classes that depend on it)."""
        graph = self.get_dependency_graph()
        reverse: dict[str, set[str]] = defaultdict(set)
        
        for source, targets in graph.items():
            for target in targets:
                reverse[target].add(source)
        
        return dict(reverse)
