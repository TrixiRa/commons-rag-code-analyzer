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
Architecture Analysis Agent for Java Codebases.

This is a lightweight wrapper that orchestrates the architecture analysis
workflow using the refactored analysis modules.
"""

from pathlib import Path
from typing import Optional, Any

from config import REPO_ROOT
from utils.logging_config import get_logger

logger = get_logger(__name__)


class ArchitectureAgent:
    """
    Agent-like workflow for architecture analysis and recommendations.
    
    Workflow:
    1. Analyze dependencies using static analysis
    2. Detect architecture issues
    3. Generate recommendations
    4. Optionally enrich with RAG context
    """
    
    def __init__(
        self,
        repo_root: Path = REPO_ROOT,
        rag_pipeline: Optional[Any] = None
    ) -> None:
        """
        Initialize the architecture agent.
        
        Args:
            repo_root: Path to the repository root
            rag_pipeline: Optional RAG pipeline for context enrichment
        """
        self.repo_root = repo_root
        self.rag_pipeline = rag_pipeline
        self._analyzer: Optional[Any] = None
        self._issues: list[Any] = []
        self._recommendations: list[Any] = []
    
    def analyze(self) -> dict[str, Any]:
        """
        Run the full architecture analysis workflow.
        
        Returns:
            Dict with summary, issues, recommendations, and metrics
        """
        from analysis.dependency_analyzer import DependencyAnalyzer
        from analysis.issue_detector import ArchitectureIssueDetector
        from analysis.refactoring_advisor import RefactoringAdvisor
        
        logger.info("=" * 60)
        logger.info("ARCHITECTURE ANALYSIS")
        logger.info("=" * 60)
        
        # Step 1: Dependency Analysis
        logger.info("Step 1/3: Analyzing dependencies...")
        self._analyzer = DependencyAnalyzer(self.repo_root)
        classes = self._analyzer.analyze()
        
        # Step 2: Issue Detection
        logger.info("Step 2/3: Detecting architecture issues...")
        detector = ArchitectureIssueDetector(self._analyzer)
        self._issues = detector.detect_all_issues()
        logger.info(f"Found {len(self._issues)} potential issues")
        
        # Step 3: Generate Recommendations
        logger.info("Step 3/3: Generating recommendations...")
        advisor = RefactoringAdvisor(self._issues, self._analyzer)
        self._recommendations = advisor.generate_recommendations()
        logger.info(f"Generated {len(self._recommendations)} recommendations")
        
        # Compute metrics
        metrics = self._compute_metrics(classes)
        
        return {
            "summary": self._generate_summary(metrics),
            "issues": self._issues,
            "recommendations": self._recommendations,
            "metrics": metrics
        }
    
    def _compute_metrics(self, classes: dict[str, Any]) -> dict[str, Any]:
        """Compute overall codebase metrics."""
        if not classes:
            return {}
        
        total_loc = sum(c.lines_of_code for c in classes.values())
        total_methods = sum(c.method_count for c in classes.values())
        packages = set(c.package for c in classes.values())
        
        avg_deps = sum(
            len(c.internal_deps) + len(c.external_deps)
            for c in classes.values()
        ) / len(classes)
        
        return {
            "total_classes": len(classes),
            "total_packages": len(packages),
            "total_lines_of_code": total_loc,
            "total_methods": total_methods,
            "avg_loc_per_class": total_loc / len(classes),
            "avg_methods_per_class": total_methods / len(classes),
            "avg_dependencies_per_class": avg_deps
        }
    
    def _generate_summary(self, metrics: dict[str, Any]) -> str:
        """Generate a human-readable summary."""
        high_issues = len([i for i in self._issues if i.severity == "high"])
        medium_issues = len([i for i in self._issues if i.severity == "medium"])
        low_issues = len([i for i in self._issues if i.severity == "low"])
        
        return f"""
Analyzed {metrics.get('total_classes', 0)} classes across {metrics.get('total_packages', 0)} packages.

CODEBASE METRICS:
- Total lines of code: {metrics.get('total_lines_of_code', 0):,}
- Average LOC per class: {metrics.get('avg_loc_per_class', 0):.1f}
- Average methods per class: {metrics.get('avg_methods_per_class', 0):.1f}
- Average dependencies per class: {metrics.get('avg_dependencies_per_class', 0):.1f}

ISSUES FOUND:
- High severity: {high_issues}
- Medium severity: {medium_issues}
- Low severity: {low_issues}

RECOMMENDATIONS: {len(self._recommendations)} actionable suggestions generated.
""".strip()
    
    def print_report(
        self,
        result: dict[str, Any],
        max_issues: int = 5,
        max_recommendations: int = 5
    ) -> None:
        """Print a formatted report of the analysis."""
        print("\n" + "=" * 60)
        print("ARCHITECTURE ANALYSIS REPORT")
        print("=" * 60)
        
        print("\n" + result["summary"])
        
        if result["issues"]:
            print("\n" + "-" * 60)
            print(f"TOP ISSUES (showing {min(max_issues, len(result['issues']))} of {len(result['issues'])})")
            print("-" * 60)
            
            sorted_issues = sorted(
                result["issues"],
                key=lambda i: {"high": 0, "medium": 1, "low": 2}[i.severity]
            )
            
            for i, issue in enumerate(sorted_issues[:max_issues], 1):
                print(f"\n{i}. [{issue.severity.upper()}] {issue.title}")
                print(f"   {issue.description}")
                files_str = ', '.join(issue.affected_files[:3])
                if len(issue.affected_files) > 3:
                    files_str += '...'
                print(f"   Files: {files_str}")
                
                if issue.issue_type == "cyclic_dependency" and issue.evidence.get("critical_imports"):
                    print("   Critical imports creating the cycle:")
                    for ci in issue.evidence["critical_imports"]:
                        print(f"     → {ci['file']}:L{ci['line']} - import {ci['import']}")
                
                if issue.issue_type == "oversized_module":
                    loc = issue.evidence.get("lines_of_code", 0)
                    methods = issue.evidence.get("method_count", 0)
                    print(f"   Metrics: {loc:,} total lines, {methods} methods")
        
        if result["recommendations"]:
            print("\n" + "-" * 60)
            print(f"TOP RECOMMENDATIONS (showing {min(max_recommendations, len(result['recommendations']))} of {len(result['recommendations'])})")
            print("-" * 60)
            
            # Separate critical (high severity) from improvement (medium/low) recommendations
            critical_recs = [r for r in result["recommendations"] if getattr(r, 'source_severity', 'medium') == 'high']
            improvement_recs = [r for r in result["recommendations"] if getattr(r, 'source_severity', 'medium') != 'high']
            
            rec_num = 1
            shown = 0
            
            # Print critical recommendations first
            if critical_recs:
                print("\n🔴 CRITICAL: Address these first to resolve major architecture issues")
                print("-" * 40)
                
                for rec in critical_recs:
                    if shown >= max_recommendations:
                        break
                    print(f"\n{rec_num}. [CRITICAL] {rec.title}")
                    print(f"   {rec.description}")
                    print(f"   Rationale: {rec.rationale}")
                    print(f"   Effort: {rec.effort}")
                    print("   Impact:")
                    for attr, impact in rec.quality_impact.items():
                        print(f"     - {attr}: {impact}")
                    if rec.concrete_examples:
                        print("   Concrete Examples:")
                        for example in rec.concrete_examples:
                            print(f"     • {example}")
                    rec_num += 1
                    shown += 1
                
                # Separator after critical issues
                if improvement_recs and shown < max_recommendations:
                    print("\n" + "=" * 60)
                    print("✅ Critical issues addressed above.")
                    print("   The following are GOOD PRACTICES for further improvement:")
                    print("=" * 60)
            
            # Print improvement recommendations
            for rec in improvement_recs:
                if shown >= max_recommendations:
                    break
                severity_tag = getattr(rec, 'source_severity', 'medium').upper()
                print(f"\n{rec_num}. [{severity_tag}] {rec.title}")
                print(f"   {rec.description}")
                print(f"   Rationale: {rec.rationale}")
                print(f"   Effort: {rec.effort}")
                print("   Impact:")
                for attr, impact in rec.quality_impact.items():
                    print(f"     - {attr}: {impact}")
                if rec.concrete_examples:
                    print("   Concrete Examples:")
                    for example in rec.concrete_examples:
                        print(f"     • {example}")
                rec_num += 1
                shown += 1
        
        print("\n" + "=" * 60)
    
    def enrich_with_rag(self, issue: Any) -> Optional[str]:
        """Use RAG to get additional context about an issue."""
        if not self.rag_pipeline:
            return None
        
        query = f"What is the purpose of {issue.affected_files[0] if issue.affected_files else 'this class'}?"
        result = self.rag_pipeline.query(query, top_k=3)
        
        if not result.uncertainty:
            return result.answer
        
        return None


def analyze_architecture(
    repo_root: Path = REPO_ROOT,
    rag_pipeline: Optional[Any] = None,
    max_issues: int = 50,
    max_recommendations: int = 15
) -> dict[str, Any]:
    """
    Convenience function to run architecture analysis.
    
    Args:
        repo_root: Path to the repository root
        rag_pipeline: Optional RAG pipeline for context enrichment
        max_issues: Maximum number of issues to display
        max_recommendations: Maximum number of recommendations to display
    
    Returns:
        Analysis results dictionary
    """
    agent = ArchitectureAgent(repo_root, rag_pipeline)
    result = agent.analyze()
    agent.print_report(result, max_issues=max_issues, max_recommendations=max_recommendations)
    return result
