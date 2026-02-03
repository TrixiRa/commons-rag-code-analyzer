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
