"""
Architecture Analysis Package

Provides static analysis tools for Java codebases to detect
architecture issues and generate refactoring recommendations.
"""

from analysis.models import DependencyInfo, ArchitectureIssue, RefactoringRecommendation
from analysis.dependency_analyzer import DependencyAnalyzer
from analysis.issue_detector import ArchitectureIssueDetector
from analysis.refactoring_advisor import RefactoringAdvisor

__all__ = [
    "DependencyInfo",
    "ArchitectureIssue", 
    "RefactoringRecommendation",
    "DependencyAnalyzer",
    "ArchitectureIssueDetector",
    "RefactoringAdvisor",
]
