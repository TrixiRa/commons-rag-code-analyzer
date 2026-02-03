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
Refactoring Advisor

Generates concrete refactoring recommendations based on detected issues.
"""

from typing import Optional

from analysis.models import ArchitectureIssue, RefactoringRecommendation, DependencyInfo
from analysis.dependency_analyzer import DependencyAnalyzer
from utils.logging_config import get_logger

logger = get_logger(__name__)


class RefactoringAdvisor:
    """Generates refactoring recommendations based on detected issues."""
    
    def __init__(
        self, 
        issues: list[ArchitectureIssue], 
        analyzer: DependencyAnalyzer
    ) -> None:
        """
        Initialize the refactoring advisor.
        
        Args:
            issues: List of detected architecture issues
            analyzer: DependencyAnalyzer with completed analysis
        """
        self.issues = issues
        self.analyzer = analyzer
        self.classes = analyzer.classes
    
    def generate_recommendations(self) -> list[RefactoringRecommendation]:
        """
        Generate refactoring recommendations for detected issues.
        
        Recommendations are sorted by:
        1. Issue severity (high → medium → low)
        2. Effort within same severity (low effort first)
        """
        recommendations: list[RefactoringRecommendation] = []
        
        # Group issues by severity
        severity_order = {"high": 0, "medium": 1, "low": 2}
        sorted_issues = sorted(
            self.issues, 
            key=lambda i: severity_order.get(i.severity, 1)
        )
        
        for issue in sorted_issues:
            rec = self._recommend_for_issue(issue)
            if rec:
                # Tag recommendation with source issue severity
                rec.source_severity = issue.severity
                recommendations.append(rec)
        
        # Sort: first by severity, then by effort within severity
        effort_order = {"low": 0, "medium": 1, "high": 2}
        recommendations.sort(
            key=lambda r: (
                severity_order.get(getattr(r, 'source_severity', 'medium'), 1),
                effort_order.get(r.effort, 1)
            )
        )
        
        return recommendations
    
    def get_critical_recommendations(self) -> list[RefactoringRecommendation]:
        """Get only recommendations for high-severity issues."""
        all_recs = self.generate_recommendations()
        return [r for r in all_recs if getattr(r, 'source_severity', 'medium') == 'high']
    
    def get_improvement_recommendations(self) -> list[RefactoringRecommendation]:
        """Get recommendations for medium/low severity issues (good practices)."""
        all_recs = self.generate_recommendations()
        return [r for r in all_recs if getattr(r, 'source_severity', 'medium') != 'high']
    
    def _recommend_for_issue(
        self, 
        issue: ArchitectureIssue
    ) -> Optional[RefactoringRecommendation]:
        """Generate a recommendation for a specific issue."""
        handlers = {
            "dependency_magnet": self._recommend_for_dependency_magnet,
            "cyclic_dependency": self._recommend_for_cyclic_dependency,
            "oversized_module": self._recommend_for_oversized_module,
            "god_class": self._recommend_for_god_class,
            "unclear_separation": self._recommend_for_unclear_separation,
        }
        
        handler = handlers.get(issue.issue_type)
        if handler:
            return handler(issue)
        return None
    
    def _recommend_for_dependency_magnet(
        self, 
        issue: ArchitectureIssue
    ) -> RefactoringRecommendation:
        """Generate recommendation for dependency magnet issue."""
        # Extract just the class name from the title
        class_name = issue.title.split(": ")[-1] if ": " in issue.title else issue.title
        dependent_count = issue.evidence["dependent_count"]
        dependents = issue.evidence.get("dependents", [])
        
        examples: list[str] = []
        examples.append(f"File: {issue.affected_files[0] if issue.affected_files else 'unknown'}")
        if dependents:
            dep_names = [d.split('.')[-1] for d in dependents[:5]]
            examples.append(f"Classes depending on {class_name}: {', '.join(dep_names)}")
            examples.append(f"Suggested interface name: I{class_name} or {class_name}Provider")
        
        return RefactoringRecommendation(
            title=f"Extract Interface for {class_name}",
            description=(
                f"Create an interface that defines the contract for {class_name}. "
                f"Have dependents rely on the interface rather than the concrete class."
            ),
            rationale=(
                f"{class_name} is depended upon by {dependent_count} classes. "
                f"An interface would decouple consumers from the implementation."
            ),
            affected_files=issue.affected_files[:5],
            quality_impact={
                "maintainability": "improved - changes don't affect dependents",
                "testability": "improved - dependents can use mocks",
                "evolvability": "improved - can swap implementations"
            },
            effort="medium",
            concrete_examples=examples
        )
    
    def _recommend_for_cyclic_dependency(
        self, 
        issue: ArchitectureIssue
    ) -> RefactoringRecommendation:
        """Generate recommendation for cyclic dependency issue."""
        cycle = issue.evidence["cycle"]
        cycle_classes = [c.split('.')[-1] for c in cycle[:-1]]
        critical_imports = issue.evidence.get("critical_imports", [])
        
        examples: list[str] = []
        for ci in critical_imports:
            examples.append(f"{ci['file']}:L{ci['line']} imports {ci['import']}")
        
        if len(cycle_classes) >= 2:
            examples.append(
                f"Suggested: Create interface I{cycle_classes[0]}Consumer "
                f"in {cycle_classes[1]}'s package"
            )
        
        return RefactoringRecommendation(
            title=f"Break Cycle: {' ↔ '.join(cycle_classes[:3])}{'...' if len(cycle_classes) > 3 else ''}",
            description=(
                f"Introduce a new interface or extract shared functionality "
                f"to break the dependency cycle."
            ),
            rationale=(
                f"Cyclic dependencies between {', '.join(cycle_classes)} create tight coupling."
            ),
            affected_files=issue.affected_files,
            quality_impact={
                "maintainability": "significantly improved",
                "testability": "significantly improved",
                "build_time": "potentially improved"
            },
            effort="high",
            concrete_examples=examples
        )
    
    def _recommend_for_oversized_module(
        self, 
        issue: ArchitectureIssue
    ) -> RefactoringRecommendation:
        """Generate recommendation for oversized module issue."""
        # Extract just the class name from the title
        class_name = issue.title.split(": ")[-1] if ": " in issue.title else issue.title
        loc = issue.evidence["lines_of_code"]
        methods = issue.evidence["method_count"]
        file_path = issue.affected_files[0] if issue.affected_files else "unknown"
        
        examples: list[str] = [
            f"File: {file_path} ({loc:,} lines, {methods} methods)",
            "Suggested extraction targets:",
            f"  - Method name prefixes (e.g., append* → {class_name}Appender)",
            f"  - Methods operating on same fields → helper class",
            f"  - Static utility methods → {class_name}Utils",
        ]
        if methods > 50:
            examples.append(f"  - Builder pattern: {class_name}Builder")
        
        return RefactoringRecommendation(
            title=f"Decompose {class_name}",
            description=(
                f"Identify cohesive groups of methods within {class_name} "
                f"and extract them into separate, focused classes."
            ),
            rationale=(
                f"At {loc:,} lines and {methods} methods, {class_name} likely "
                f"violates the Single Responsibility Principle."
            ),
            affected_files=issue.affected_files,
            quality_impact={
                "maintainability": "improved - smaller classes",
                "testability": "improved - focused classes",
                "reusability": "improved - extracted classes may be reusable"
            },
            effort="high",
            concrete_examples=examples
        )
    
    def _recommend_for_god_class(
        self, 
        issue: ArchitectureIssue
    ) -> RefactoringRecommendation:
        """Generate recommendation for god class issue."""
        # Extract just the class name from the title
        class_name = issue.title.split(": ")[-1] if ": " in issue.title else issue.title
        total_deps = issue.evidence["total_dependencies"]
        internal_deps = issue.evidence.get("internal_deps", [])
        external_deps = issue.evidence.get("external_deps", [])
        file_path = issue.affected_files[0] if issue.affected_files else "unknown"
        
        examples: list[str] = [
            f"File: {file_path}",
            f"Total dependencies: {total_deps} ({len(internal_deps)} internal, {len(external_deps)} external)",
        ]
        if internal_deps:
            examples.append(f"Internal: {', '.join([d.split('.')[-1] for d in internal_deps[:5]])}")
        if external_deps:
            examples.append(f"External: {', '.join([d.split('.')[-1] for d in external_deps[:5]])}")
        
        return RefactoringRecommendation(
            title=f"Reduce Coupling in {class_name}",
            description=(
                f"Review the {total_deps} dependencies. Consider dependency injection, "
                f"extracting helper classes, or using facades."
            ),
            rationale=(
                f"High coupling makes {class_name} difficult to test and maintain."
            ),
            affected_files=issue.affected_files,
            quality_impact={
                "maintainability": "improved - fewer reasons to change",
                "testability": "improved - easier to mock",
                "understandability": "improved - clearer purpose"
            },
            effort="medium",
            concrete_examples=examples
        )
    
    def _recommend_for_unclear_separation(
        self, 
        issue: ArchitectureIssue
    ) -> RefactoringRecommendation:
        """Generate recommendation for unclear separation issue."""
        source = issue.evidence["source_package"].split('.')[-1]
        target = issue.evidence["target_package"].split('.')[-1]
        dep_count = issue.evidence.get("dependency_count", 0)
        
        examples: list[str] = [
            f"Source package: {issue.evidence['source_package']}",
            f"Target package: {issue.evidence['target_package']}",
            f"Cross-package dependencies: {dep_count}",
            f"Suggested: Create {target.capitalize()}Api or {target.capitalize()}Facade",
        ]
        
        return RefactoringRecommendation(
            title=f"Clarify Boundary: {source} / {target}",
            description=(
                f"Define a clear interface between {source} and {target} packages."
            ),
            rationale=(
                f"Multiple cross-package dependencies suggest unclear boundaries."
            ),
            affected_files=issue.affected_files,
            quality_impact={
                "maintainability": "improved - clearer responsibilities",
                "evolvability": "improved - independent evolution",
                "understandability": "improved - explicit architecture"
            },
            effort="low",
            concrete_examples=examples
        )
