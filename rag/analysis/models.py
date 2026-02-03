"""
Data Models for Architecture Analysis

Contains all dataclasses used across the architecture analysis modules.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DependencyInfo:
    """Information about a class and its dependencies."""
    path: str
    package: str
    class_name: str
    imports: list[str] = field(default_factory=list)
    import_lines: dict[str, int] = field(default_factory=dict)  # Map import -> line number
    internal_deps: list[str] = field(default_factory=list)  # Dependencies within the project
    external_deps: list[str] = field(default_factory=list)  # External library dependencies
    lines_of_code: int = 0  # Total lines in file
    lines_of_code_nonblank: int = 0  # Non-blank, non-comment lines
    method_count: int = 0


@dataclass
class ArchitectureIssue:
    """Represents an identified architecture issue."""
    issue_type: str  # e.g., "dependency_magnet", "cyclic_dependency", "oversized_module"
    severity: str    # "high", "medium", "low"
    title: str
    description: str
    affected_files: list[str]
    evidence: dict[str, Any]   # Metrics/data supporting the issue


@dataclass
class RefactoringRecommendation:
    """A concrete refactoring or architectural improvement suggestion."""
    title: str
    description: str
    rationale: str
    affected_files: list[str]
    quality_impact: dict[str, str]  # e.g., {"maintainability": "+", "testability": "+"}
    effort: str  # "low", "medium", "high"
    concrete_examples: list[str] = field(default_factory=list)  # Specific code examples


@dataclass
class AnalysisResult:
    """Complete result of architecture analysis."""
    summary: str
    issues: list[ArchitectureIssue]
    recommendations: list[RefactoringRecommendation]
    metrics: dict[str, Any]
