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
Architecture Issue Detector

Detects various architecture issues and code smells in Java codebases.
Uses percentile-based thresholds to flag outliers relative to the codebase.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional, Union
import statistics

from analysis.models import DependencyInfo, ArchitectureIssue
from analysis.dependency_analyzer import DependencyAnalyzer
from utils.logging_config import get_logger

logger = get_logger(__name__)


# Percentile thresholds - flag classes in the top N percentile
# E.g., 90 means top 10% are flagged
LOC_PERCENTILE = 90  # Top 10% by lines of code
METHOD_PERCENTILE = 90  # Top 10% by method count
DEPENDENCY_PERCENTILE = 85  # Top 15% by dependency count
DEPENDENT_PERCENTILE = 85  # Top 15% by number of dependents

# Other constants
CYCLIC_DEPENDENCY_MAX_DISPLAY = 5  # Max cycles to show
CROSS_PACKAGE_DEP_THRESHOLD = 5  # Minimum cross-package deps to report


def compute_percentile(values: list[float], percentile: float) -> float:
    """Compute the percentile value from a list."""
    if not values:
        return 0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    idx = (percentile / 100) * (n - 1)
    lower = int(idx)
    upper = min(lower + 1, n - 1)
    fraction = idx - lower
    return sorted_vals[lower] + fraction * (sorted_vals[upper] - sorted_vals[lower])


class ArchitectureIssueDetector:
    """
    Detects architecture issues and code smells.
    
    Uses percentile-based thresholds to identify outliers relative to the
    codebase distribution, rather than hardcoded values.
    """
    
    def __init__(self, analyzer: DependencyAnalyzer) -> None:
        """
        Initialize the issue detector.
        
        Args:
            analyzer: DependencyAnalyzer with completed analysis
        """
        self.analyzer = analyzer
        self.classes = analyzer.classes
        self.dep_graph = analyzer.get_dependency_graph()
        self.reverse_dep_graph = analyzer.get_reverse_dependency_graph()
        
        # Compute distribution statistics once
        self._compute_statistics()
    
    def _compute_statistics(self) -> None:
        """Compute percentile thresholds based on codebase distribution."""
        locs = [info.lines_of_code for info in self.classes.values()]
        methods = [info.method_count for info in self.classes.values()]
        deps = [
            len(info.internal_deps) + len(info.external_deps) 
            for info in self.classes.values()
        ]
        dependents = [
            len(self.reverse_dep_graph.get(fqn, set()))
            for fqn in self.classes
        ]
        
        # Compute thresholds
        self.loc_threshold = compute_percentile(locs, LOC_PERCENTILE)
        self.method_threshold = compute_percentile(methods, METHOD_PERCENTILE)
        self.dep_threshold = compute_percentile(deps, DEPENDENCY_PERCENTILE)
        self.dependent_threshold = compute_percentile(dependents, DEPENDENT_PERCENTILE)
        
        # Store stats for reporting
        self.stats = {
            "loc": {
                "min": min(locs) if locs else 0,
                "max": max(locs) if locs else 0,
                "mean": statistics.mean(locs) if locs else 0,
                "median": statistics.median(locs) if locs else 0,
                "p90_threshold": self.loc_threshold,
            },
            "methods": {
                "min": min(methods) if methods else 0,
                "max": max(methods) if methods else 0,
                "mean": statistics.mean(methods) if methods else 0,
                "median": statistics.median(methods) if methods else 0,
                "p90_threshold": self.method_threshold,
            },
            "dependencies": {
                "min": min(deps) if deps else 0,
                "max": max(deps) if deps else 0,
                "mean": statistics.mean(deps) if deps else 0,
                "median": statistics.median(deps) if deps else 0,
                "p85_threshold": self.dep_threshold,
            },
            "dependents": {
                "min": min(dependents) if dependents else 0,
                "max": max(dependents) if dependents else 0,
                "mean": statistics.mean(dependents) if dependents else 0,
                "median": statistics.median(dependents) if dependents else 0,
                "p85_threshold": self.dependent_threshold,
            },
        }
    
    def get_statistics(self) -> dict:
        """Return computed statistics for display."""
        return self.stats
    
    def detect_all_issues(self) -> list[ArchitectureIssue]:
        """Run all detection methods and return found issues."""
        issues: list[ArchitectureIssue] = []
        issues.extend(self.detect_dependency_magnets())
        issues.extend(self.detect_cyclic_dependencies())
        issues.extend(self.detect_oversized_modules())
        issues.extend(self.detect_god_classes())
        issues.extend(self.detect_unclear_separation())
        return issues
    
    def detect_dependency_magnets(self) -> list[ArchitectureIssue]:
        """
        Find classes that have too many dependents (top percentile).
        
        Returns:
            List of dependency magnet issues
        """
        issues: list[ArchitectureIssue] = []
        
        for fqn, dependents in self.reverse_dep_graph.items():
            dep_count = len(dependents)
            if dep_count >= self.dependent_threshold and fqn in self.classes:
                info = self.classes[fqn]
                issues.append(ArchitectureIssue(
                    issue_type="dependency_magnet",
                    severity="medium" if dep_count < 10 else "high",
                    title=f"Dependency Accumulation: {info.class_name}",
                    description=(
                        f"The class `{info.class_name}` is depended upon by {dep_count} "
                        f"other class(es) (top {100 - DEPENDENT_PERCENTILE:.0f}% in the codebase). "
                        f"Changes to this class may have wide-ranging impact."
                    ),
                    affected_files=[info.path] + [
                        self.classes[d].path for d in dependents if d in self.classes
                    ],
                    evidence={
                        "dependent_count": dep_count,
                        "dependents": list(dependents)[:10],
                        "class_loc": info.lines_of_code,
                        "method_count": info.method_count,
                        "percentile_threshold": self.dependent_threshold,
                        "codebase_median": self.stats["dependents"]["median"],
                        "codebase_max": self.stats["dependents"]["max"]
                    }
                ))
        
        return issues
    
    def detect_cyclic_dependencies(self) -> list[ArchitectureIssue]:
        """Detect circular dependencies between classes."""
        issues: list[ArchitectureIssue] = []
        visited: set[str] = set()
        
        def find_cycles(node: str, path: list[str]) -> list[list[str]]:
            cycles: list[list[str]] = []
            if node in path:
                cycle_start = path.index(node)
                cycles.append(path[cycle_start:] + [node])
                return cycles
            
            if node in visited or node not in self.dep_graph:
                return cycles
            
            visited.add(node)
            for dep in self.dep_graph.get(node, []):
                cycles.extend(find_cycles(dep, path + [node]))
            
            return cycles
        
        all_cycles: list[list[str]] = []
        for fqn in self.classes:
            visited.clear()
            cycles = find_cycles(fqn, [])
            for cycle in cycles:
                normalized = tuple(sorted(cycle[:-1]))
                if normalized not in [tuple(sorted(c[:-1])) for c in all_cycles]:
                    all_cycles.append(cycle)
        
        for cycle in all_cycles[:CYCLIC_DEPENDENCY_MAX_DISPLAY]:
            cycle_classes = [c.split('.')[-1] for c in cycle]
            affected = [self.classes[c].path for c in cycle[:-1] if c in self.classes]
            
            critical_imports = self._find_critical_imports(cycle)
            
            issues.append(ArchitectureIssue(
                issue_type="cyclic_dependency",
                severity="high",
                title=f"Cyclic Dependency: {' → '.join(cycle_classes)}",
                description=(
                    f"A circular dependency exists between these classes: "
                    f"{' → '.join(cycle_classes)}."
                ),
                affected_files=affected,
                evidence={
                    "cycle": cycle,
                    "cycle_length": len(cycle) - 1,
                    "critical_imports": critical_imports
                }
            ))
        
        return issues
    
    def _find_critical_imports(self, cycle: list[str]) -> list[dict[str, str | int]]:
        """Find the import statements that create a dependency cycle."""
        critical_imports: list[dict[str, str | int]] = []
        
        for i in range(len(cycle) - 1):
            source_fqn = cycle[i]
            target_fqn = cycle[i + 1]
            if source_fqn in self.classes:
                source_info = self.classes[source_fqn]
                target_class = target_fqn.split('.')[-1]
                for imp, line_num in source_info.import_lines.items():
                    if target_class in imp or imp == target_fqn:
                        critical_imports.append({
                            "file": source_info.path,
                            "line": line_num,
                            "import": imp,
                            "creates_cycle_to": target_class
                        })
                        break
        
        return critical_imports
    
    def detect_oversized_modules(self) -> list[ArchitectureIssue]:
        """Find classes that are too large (top percentile by LOC or methods)."""
        issues: list[ArchitectureIssue] = []
        
        for fqn, info in self.classes.items():
            loc_exceeds = info.lines_of_code > self.loc_threshold
            method_exceeds = info.method_count > self.method_threshold
            
            if loc_exceeds or method_exceeds:
                # Severity based on how extreme the outlier is
                loc_ratio = info.lines_of_code / self.loc_threshold if self.loc_threshold > 0 else 1
                method_ratio = info.method_count / self.method_threshold if self.method_threshold > 0 else 1
                severity = "high" if (loc_ratio > 2 or method_ratio > 2) else "medium"
                
                issues.append(ArchitectureIssue(
                    issue_type="oversized_module",
                    severity=severity,
                    title=f"Large (potentially oversized) Class: {info.class_name}",
                    description=(
                        f"The class `{info.class_name}` has {info.lines_of_code} lines of code "
                        f"and {info.method_count} methods (top {100 - LOC_PERCENTILE:.0f}% in codebase)."
                    ),
                    affected_files=[info.path],
                    evidence={
                        "lines_of_code": info.lines_of_code,
                        "method_count": info.method_count,
                        "loc_percentile_threshold": self.loc_threshold,
                        "method_percentile_threshold": self.method_threshold,
                        "codebase_median_loc": self.stats["loc"]["median"],
                        "codebase_max_loc": self.stats["loc"]["max"],
                        "codebase_median_methods": self.stats["methods"]["median"],
                        "codebase_max_methods": self.stats["methods"]["max"]
                    }
                ))
        
        return issues
    
    def detect_god_classes(self) -> list[ArchitectureIssue]:
        """Find classes with too many dependencies (top percentile)."""
        issues: list[ArchitectureIssue] = []
        
        for fqn, info in self.classes.items():
            total_deps = len(info.internal_deps) + len(info.external_deps)
            if total_deps > self.dep_threshold:
                dep_ratio = total_deps / self.dep_threshold if self.dep_threshold > 0 else 1
                severity = "high" if dep_ratio > 1.5 else "medium"
                
                issues.append(ArchitectureIssue(
                    issue_type="god_class",
                    severity=severity,
                    title=f"Coupling: {info.class_name}",
                    description=(
                        f"The class `{info.class_name}` has {total_deps} dependencies "
                        f"({len(info.internal_deps)} internal, {len(info.external_deps)} external) - "
                        f"corresponding to top {100 - DEPENDENCY_PERCENTILE:.0f}% in the codebase."
                    ),
                    affected_files=[info.path],
                    evidence={
                        "total_dependencies": total_deps,
                        "internal_deps": info.internal_deps[:10],
                        "external_deps": info.external_deps[:10],
                        "internal_dep_count": len(info.internal_deps),
                        "external_dep_count": len(info.external_deps),
                        "percentile_threshold": self.dep_threshold,
                        "codebase_median": self.stats["dependencies"]["median"],
                        "codebase_max": self.stats["dependencies"]["max"]
                    }
                ))
        
        return issues
    
    def detect_unclear_separation(self) -> list[ArchitectureIssue]:
        """Detect potential separation of concerns issues."""
        issues: list[ArchitectureIssue] = []
        
        packages: dict[str, list[str]] = defaultdict(list)
        for fqn, info in self.classes.items():
            packages[info.package].append(fqn)
        
        cross_package_deps: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        
        for fqn, deps in self.dep_graph.items():
            if fqn not in self.classes:
                continue
            source_pkg = self.classes[fqn].package
            
            for dep in deps:
                if dep in self.classes:
                    target_pkg = self.classes[dep].package
                    if source_pkg != target_pkg:
                        cross_package_deps[source_pkg][target_pkg] += 1
        
        for source_pkg, targets in cross_package_deps.items():
            for target_pkg, count in targets.items():
                if count >= CROSS_PACKAGE_DEP_THRESHOLD:
                    source_subpkg = source_pkg.replace(
                        self.analyzer.package_prefix + ".", ""
                    )
                    target_subpkg = target_pkg.replace(
                        self.analyzer.package_prefix + ".", ""
                    )
                    
                    if source_subpkg and target_subpkg and source_subpkg != target_subpkg:
                        affected = [
                            self.classes[c].path 
                            for c in packages[source_pkg] 
                            if c in self.classes
                        ]
                        
                        issues.append(ArchitectureIssue(
                            issue_type="unclear_separation",
                            severity="low",
                            title=f"Cross-Package Coupling: {source_subpkg} → {target_subpkg}",
                            description=(
                                f"The package `{source_subpkg}` has {count} dependencies "
                                f"on `{target_subpkg}`."
                            ),
                            affected_files=affected[:5],
                            evidence={
                                "source_package": source_pkg,
                                "target_package": target_pkg,
                                "dependency_count": count
                            }
                        ))
        
        return issues
