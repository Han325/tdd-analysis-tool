import re
from typing import Dict, Tuple, List, NamedTuple, Set, Optional
from dataclasses import dataclass
from datetime import datetime
import os
from collections import defaultdict
from git import Repo, Commit
from pydriller import Repository, ModificationType
import pandas as pd
from datetime import datetime
import logging
import ast
import javalang
from javalang.parser import Parser
from javalang.tree import MethodDeclaration, ClassDeclaration
import networkx as nx

# Create logs directory if it doesn't exist
if not os.path.exists("logs"):
    os.makedirs("logs")

# Create timestamped log filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join("logs", f"tdd_analysis_{timestamp}.log")

# Configure logging (DISABLED)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
)

# List of repository URLs
REPO_URLS = [
    # "https://github.com/apache/bigtop-manager",
    # "https://github.com/apache/commons-csv",
    "https://github.com/apache/doris-kafka-connector"
]

# Directory to clone repositories
CLONE_DIR = "apache_repos"

# Directory to save analysis results
OUTPUT_DIR = "debug_results"

# Enhanced patterns for test detection
TEST_PATTERNS = {
    "java": {
        "file_patterns": [
            r"Test.*\.java$",
            r".*Test\.java$",
            r".*Tests\.java$",
            r".*TestCase\.java$",
            r".*IT\.java$",  # Integration tests
            r".*ITCase\.java$",
        ],
        "frameworks": [
            "org.junit",
            "org.testng",
            "org.mockito",
            "org.easymock",
            "org.powermock",
        ],
        "annotations": ["@Test", "@Before", "@After", "@BeforeClass", "@AfterClass"],
    }
}


@dataclass
class FileContent:
    """Represents parsed content of a file at a specific point in time"""
    raw_content: str
    imports: List[str]
    class_names: List[str]
    method_names: List[str]
    is_abstract: bool
    is_utility: bool
    test_frameworks: List[str]
    test_methods: List[str]
    dependencies: List[str]
    
    @property
    def is_test(self) -> bool:
        """Determine if file is a test based on content analysis"""
        return bool(self.test_frameworks and self.test_methods and not self.is_abstract)


@dataclass
class FileHistory:
    """Tracks a file's history through version control"""
    creation_date: datetime
    full_path: str
    basename: str
    directory: str
    last_modified_date: datetime
    creation_commit: str
    modifications: List[Tuple[datetime, str]]  # List of (date, commit_hash)
    content_history: List[Tuple[datetime, FileContent]]
    related_files: Set[str]  # Set of related file paths
    is_deleted: bool = False

    @property
    def latest_content(self) -> Optional[FileContent]:
        """Get the most recent content version"""
        return self.content_history[-1][1] if self.content_history else None

    @property
    def is_test(self) -> bool:
        """Determine if file is a test based on latest content"""
        return bool(self.latest_content and self.latest_content.is_test)
    
    @property
    def is_abstract(self) -> bool:
        """Get abstract status from latest content"""
        return bool(self.latest_content and self.latest_content.is_abstract)
    
    @property
    def is_utility(self) -> bool:
        """Get utility status from latest content"""
        return bool(self.latest_content and self.latest_content.is_utility)

class MatchedPair(NamedTuple):
    test_file: FileHistory
    source_file: FileHistory
    directory_match: bool
    confidence_score: float  # Indicates confidence in the match
    relationship_type: str  # 'direct', 'abstract', 'utility', etc.


class CommitGraph:
    def __init__(self):
        self.graph = nx.DiGraph()

    def add_commit(self, commit_hash: str, parent_hashes: List[str], branch: str):
        self.graph.add_node(commit_hash, branch=branch)
        for parent in parent_hashes:
            self.graph.add_edge(parent, commit_hash)

    def find_common_ancestor(self, commit1: str, commit2: str) -> Optional[str]:
        try:
            return nx.lowest_common_ancestor(self.graph, commit1, commit2)
        except:
            return None

    def get_branch_point(self, commit_hash: str) -> Optional[str]:
        predecessors = list(self.graph.predecessors(commit_hash))
        if len(predecessors) > 1:
            return commit_hash
        return None


def clone_repos(repo_urls, clone_dir):
    if not os.path.exists(clone_dir):
        os.makedirs(clone_dir)

    for url in repo_urls:
        repo_name = url.split("/")[-1].replace(".git", "")
        repo_path = os.path.join(clone_dir, repo_name)

        if not os.path.exists(repo_path):
            try:
                Repo.clone_from(url, repo_path)
            except Exception as e:
                logging.error(f"Failed to clone {repo_name}: {str(e)}")
        else:
            logging.info(f"{repo_name} already exists at {repo_path}")


def analyze_java_content(content: str) -> FileContent:
    tree = javalang.parse.parse(content)
    imports = [imp.path for imp in tree.imports]

    class_names = []
    method_names = []
    is_abstract = False
    is_utility = False
    test_frameworks = []
    test_methods = []
    dependencies = []
    
    # First check if this is a real test class by looking for test framework imports/annotations
    has_test_framework = any(framework in imp for imp in imports for framework in TEST_PATTERNS["java"]["frameworks"])
    has_test_annotation = False

    for path, node in tree.filter(ClassDeclaration):
        class_names.append(node.name)

        # Check for abstract modifier
        if "abstract" in node.modifiers:
            is_abstract = True

        if all("static" in method.modifiers for method in node.methods):
            is_utility = True

        # Look for @Test annotations at class level
        if hasattr(node, 'annotations'):
            for annotation in node.annotations:
                if annotation.name == 'Test':
                    has_test_annotation = True

        # Only look for test methods if class is not abstract
        if not is_abstract:
            for method in node.methods:
                method_names.append(method.name)
                
                # Only consider method as test if class uses test framework
                if has_test_framework:
                    if hasattr(method, 'annotations') and method.annotations:
                        for annotation in method.annotations:
                            if annotation.name == 'Test':
                                has_test_annotation = True
                                test_methods.append(method.name)
                                break

    # Only include test frameworks if actually test class and not abstract
    if has_test_framework and has_test_annotation and not is_abstract:
        for framework in TEST_PATTERNS["java"]["frameworks"]:
            if any(framework in imp for imp in imports):
                test_frameworks.append(framework)

    # Enhanced dependency extraction
    dependencies = set()
    dependencies.update(imports)
    for path, node in tree.filter(javalang.tree.ReferenceType):
        if hasattr(node, 'name'):
            dependencies.add(node.name)

    return FileContent(
        raw_content=content,
        imports=imports,
        class_names=class_names,
        method_names=method_names,
        is_abstract=is_abstract,
        is_utility=is_utility,
        test_frameworks=test_frameworks,
        test_methods=test_methods,
        dependencies=list(dependencies)
    )


def get_file_creation_dates(
    repo_path: str,
) -> Tuple[Dict[str, FileHistory], CommitGraph]:
    file_histories: Dict[str, FileHistory] = {}
    commit_graph = CommitGraph()

    repo = Repo(repo_path)
    default_branch = repo.active_branch.name

    commit_count = 0
    for commit in Repository(
        repo_path, only_in_branch=default_branch
    ).traverse_commits():
        commit_count += 1

        # Add commit to graph
        if isinstance(commit, Commit):
            commit_graph.add_commit(
                commit.hash, [p.hash for p in commit.parents], default_branch
            )

        logging.debug(f"Processing commit {commit.hash[:8]} from {commit.author_date}")

        for modification in commit.modified_files:
            current_path = modification.new_path or modification.old_path
            if not current_path:
                logging.warning(
                    f"Skipping modification with no path in commit {commit.hash[:8]}"
                )
                continue

            normalized_path = os.path.normpath(current_path)
            basename = os.path.basename(normalized_path)
            directory = os.path.dirname(normalized_path)

            logging.debug(f"Processing file: {normalized_path}")
            logging.debug(f"Change type: {modification.change_type}")

            if modification.change_type == ModificationType.ADD:
                if normalized_path not in file_histories:
                    # Parse content based on file type
                    content = FileContent("", [], [], [], False, False, [], [], [])
                    if normalized_path.endswith(".java"):
                        content = analyze_java_content(modification.source_code)

                    file_histories[normalized_path] = FileHistory(
                        creation_date=commit.author_date,
                        full_path=normalized_path,
                        basename=basename,
                        directory=directory,
                        last_modified_date=commit.author_date,
                        creation_commit=commit.hash,
                        modifications=[(commit.author_date, commit.hash)],
                        content_history=[(commit.author_date, content)],
                        related_files=set(),
                        is_deleted=False
                    )
            elif modification.change_type == ModificationType.DELETE:
                if normalized_path in file_histories:
                    file_histories[normalized_path].is_deleted = True
                    file_histories[normalized_path].modifications.append(
                        (commit.author_date, commit.hash)
                    )

            elif modification.change_type == ModificationType.MODIFY:
                if normalized_path in file_histories:
                    file_histories[normalized_path].last_modified_date = (
                        commit.author_date
                    )
                    file_histories[normalized_path].modifications.append(
                        (commit.author_date, commit.hash)
                    )

                    # Update content history for modified files
                    content = FileContent("", [], [], [], False, False, [], [], [])
                    if normalized_path.endswith(".java"):
                        content = analyze_java_content(modification.source_code)

                    file_histories[normalized_path].content_history.append(
                        (commit.author_date, content)
                    )

    return file_histories, commit_graph


def is_related_directory(test_dir: str, source_dir: str) -> bool:
    """
    Enhanced analysis of directory relationships to handle various project structures.
    Returns True if directories are considered related, False otherwise.

    Handles:
    - Standard Maven/Gradle layout (src/test vs src/main)
    - Python package testing conventions
    - Nested module structures
    - Integration test directories
    - Custom test directory conventions
    - Monorepo structures
    """
    # Normalize paths for comparison
    test_components = [c.lower() for c in test_dir.split(os.sep) if c]
    source_components = [c.lower() for c in source_dir.split(os.sep) if c]

    # Common test-source directory patterns
    RELATED_PATTERNS = [
        # Standard Java patterns
        ("test", "main"),
        ("tests", "main"),
        ("test", "src"),
        ("tests", "src"),
        # Integration test patterns
        ("integration-test", "main"),
        ("integration-tests", "main"),
        ("it", "main"),
        # Python patterns # keeping this just in case some java projects use this
        ("tests", ""),  # Python package with tests at root
        ("test", ""),
        # Custom but common patterns
        ("unit-tests", "src"),
        ("unit", "src"),
        ("testing", "src"),
        ("testcases", "src"),
    ]

    # Check for exact directory replacement patterns
    for test_pattern, source_pattern in RELATED_PATTERNS:
        test_normalized = [
            c if c != test_pattern else source_pattern for c in test_components
        ]
        if test_normalized == source_components:
            logging.debug(
                f"Found related directories with pattern: {test_pattern} -> {source_pattern}"
            )
            return True

    # Handle monorepo and nested module structures
    if len(test_components) >= 2 and len(source_components) >= 2:
        # Check if they share the same module/project root
        module_depth = min(len(test_components), len(source_components)) - 1
        if test_components[:module_depth] == source_components[:module_depth]:
            test_suffix = test_components[module_depth:]
            source_suffix = source_components[module_depth:]

            # Check if the remaining parts match any patterns
            for test_pattern, source_pattern in RELATED_PATTERNS:
                if any(part == test_pattern for part in test_suffix) and any(
                    part == source_pattern for part in source_suffix
                ):
                    logging.debug(f"Found related directories in module structure")
                    return True

    # Handle mirror directory structures
    # e.g., com/example/service/impl vs com/example/service/test
    if len(test_components) == len(source_components):
        differences = sum(
            1 for t, s in zip(test_components, source_components) if t != s
        )
        if differences == 1 and any(
            test_marker in test_components
            for test_marker in ["test", "tests", "testing"]
        ):
            logging.debug("Found mirror directory structure with test marker")
            return True

    # Handle special case: test directory at same level
    if len(test_components) > 0 and len(source_components) > 0:
        test_parent = test_components[:-1]
        source_parent = source_components[:-1]
        if test_parent == source_parent and any(
            test_marker in test_components[-1]
            for test_marker in ["test", "tests", "testing"]
        ):
            logging.debug("Found test directory at same level as source")
            return True

    # Special case for integration tests which might be in a separate root directory
    if "integration" in test_components or "it" in test_components:
        # Remove integration test specific parts and compare the rest
        cleaned_test = [
            c
            for c in test_components
            if c not in ["integration", "it", "tests", "test"]
        ]
        cleaned_source = source_components
        if any(cs in cleaned_test for cs in cleaned_source):
            logging.debug("Found related integration test directory")
            return True

    logging.debug("No directory relationship found")
    return False


def calculate_match_confidence(
    test_file: FileHistory, source_file: FileHistory
) -> float:
    confidence = 0.0
    if test_file.basename.replace("Test", "") == source_file.basename:
        confidence += 0.4

    # Directory relationship
    if test_file.directory == source_file.directory:
        confidence += 0.2
    elif is_related_directory(test_file.directory, source_file.directory):
        confidence += 0.1

    # Content-based confidence
    latest_test_content = (
        test_file.content_history[-1][1] if test_file.content_history else None
    )
    latest_source_content = (
        source_file.content_history[-1][1] if source_file.content_history else None
    )

    if latest_test_content and latest_source_content:
        # Check if source class is referenced in test
        if source_file.basename[:-5] in " ".join(
            latest_test_content.raw_content.split()
        ):
            confidence += 0.2

        # Check for matching method names (excluding 'test' prefix)
        test_methods = set(
            m[5:] if m.startswith("test_") else m
            for m in latest_test_content.method_names
        )
        source_methods = set(latest_source_content.method_names)
        if test_methods & source_methods:  # If there's any overlap
            confidence += 0.2

    return min(1.0, confidence)

def get_base_name(filename: str) -> str:
    # Remove Test from end: UserTest.java -> User.java
    base = re.sub(r"Test[s]?\.java$", ".java", filename)
    # Remove Test from start: TestUser.java -> User.java 
    base = re.sub(r"^Test(.*)\.java$", r"\1.java", base)
    return base

def match_tests_sources(file_histories: Dict[str, FileHistory], commit_graph: CommitGraph) -> List[MatchedPair]:
    logging.info("Starting initial test-source file matching process")
    matches = []

    basename_groups = defaultdict(list)
    test_files_count = 0
    source_files_count = 0
    
    for path, history in file_histories.items():
        if not history.is_deleted:
            if history.is_test:
                test_files_count += 1
                logging.debug(f"Found test file: {path}")
            else:
                source_files_count += 1
                logging.debug(f"Found source file: {path}")
            base = get_base_name(history.basename)
            basename_groups[base].append(history)

    logging.info(f"Total test files found: {test_files_count}")
    logging.info(f"Total source files found: {source_files_count}")
    
    # Original matching logic (unchanged)
    for base, histories in basename_groups.items():
        logging.debug(f"\nProcessing base name group: {base}")
        test_files = [h for h in histories if h.is_test]
        source_files = [h for h in histories if not h.is_test]
        logging.debug(f"Found {len(test_files)} test files and {len(source_files)} source files in group")

        for test_file in test_files:
            logging.debug(f"\nAnalyzing test file: {test_file.full_path}")
            best_match = None
            best_confidence = 0.0
            relationship_type = "direct"

            # Check each potential source file
            for source_file in source_files:
                logging.debug(f"Comparing with source file: {source_file.full_path}")
                confidence = calculate_match_confidence(test_file, source_file)
                logging.debug(f"Initial confidence score: {confidence}")

                # Adjust confidence based on commit history
                if test_file.creation_commit == source_file.creation_commit:
                    # Check for same-commit TDD patterns
                    if has_tdd_indicators(test_file, source_file):
                        confidence += 0.1
                        logging.debug(f"Added TDD indicator bonus. New confidence: {confidence}")

                # Check if files were moved from another repository
                if indicates_repository_move(test_file, source_file, commit_graph):
                    confidence += 0.05
                    logging.debug(f"Added repository move bonus. New confidence: {confidence}")

                if confidence > best_confidence:
                    logging.debug(f"New best match found with confidence {confidence}")
                    best_confidence = confidence
                    best_match = source_file

                    # Determine relationship type
                    if source_file.is_abstract:
                        relationship_type = "abstract"
                        logging.debug("Relationship type: abstract")
                    elif source_file.is_utility:
                        relationship_type = "utility"
                        logging.debug("Relationship type: utility")
                    else:
                        logging.debug("Relationship type: direct")

            if best_match and best_confidence > 0.3:  # Confidence threshold
                logging.info(f"Match found: {test_file.full_path} -> {best_match.full_path}")
                logging.info(f"Confidence: {best_confidence}, Type: {relationship_type}")
                matches.append(
                    MatchedPair(
                        test_file=test_file,
                        source_file=best_match,
                        directory_match=test_file.directory == best_match.directory,
                        confidence_score=best_confidence,
                        relationship_type=relationship_type,
                    )
                )
   

    # Second pass: Check remaining unmatched test files using imports
    matched_test_paths = {match.test_file.full_path for match in matches}
    all_test_files = [h for h in file_histories.values() if h.is_test and not h.is_deleted]
    remaining_test_files = [f for f in all_test_files if f.full_path not in matched_test_paths]
    
    if remaining_test_files:
        logging.info(f"Starting import-based matching for {len(remaining_test_files)} unmatched test files")
        source_files = [h for h in file_histories.values() if not h.is_test and not h.is_deleted]
        
        for test_file in remaining_test_files:
            if test_file.content_history:
                latest_test_content = test_file.content_history[-1][1]
                best_match = None
                best_confidence = 0.0
                
                for source_file in source_files:
                    source_class = source_file.basename[:-5]
                    for import_stmt in latest_test_content.imports:
                        if source_class == import_stmt.split('.')[-1]:
                            confidence = 0.5
                            if confidence > best_confidence:
                                best_confidence = confidence
                                best_match = source_file
                
                if best_match:
                    logging.info(f"Import-based match found: {test_file.full_path} -> {best_match.full_path}")
                    matches.append(
                        MatchedPair(
                            test_file=test_file,
                            source_file=best_match,
                            directory_match=test_file.directory == best_match.directory,
                            confidence_score=best_confidence,
                            relationship_type="direct"
                        )
                    )

    return matches


def has_tdd_indicators(test_file: FileHistory, source_file: FileHistory) -> bool:
    """
    Analyzes test and source files for TDD practice indicators.
    Returns True if there are strong indicators that TDD was followed.
    """
    # Get initial content from creation time
    test_initial = next(
        (c for d, c in test_file.content_history if d == test_file.creation_date), 
        None
    )
    source_initial = next(
        (c for d, c in source_file.content_history if d == source_file.creation_date),
        None
    )

    if not (test_initial and source_initial):
        return False

    # Enhanced test completeness check
    test_methods_complete = False
    if test_initial.test_methods:
        # Get the raw content to analyze test methods
        raw_content = test_initial.raw_content
        
        for method_name in test_initial.test_methods:
            # Find method content using regex
            method_pattern = "public.*" + re.escape(method_name) + ".*?\\{([^}]*)\\}"
            method_matches = re.findall(method_pattern, raw_content, re.DOTALL)
            
            if method_matches:
                method_content = method_matches[0]  # Get the method body
                
                # Check for meaningful test implementation
                has_assertions = any(pattern in method_content.lower() 
                                   for pattern in ["assert", "expect", "should", "verify"])
                # TODO: Justify why this is the number
                has_meaningful_length = len(method_content.strip()) > 50  # Require substantial test content
                
                if has_assertions and has_meaningful_length:
                    test_methods_complete = True
                    break

    # Enhanced source method check for skeletal implementation
    source_methods_skeletal = False
    if source_initial.method_names:
        empty_or_todo_methods = 0
        total_methods = len(source_initial.method_names)
        
        # TODO: Justify why this is skeletal.
        for method_name in source_initial.method_names:
            method_pattern = ".*" + re.escape(method_name) + ".*?\\{([^}]*)\\}"
            matches = re.findall(method_pattern, source_initial.raw_content, re.DOTALL)
            
            if matches:
                method_body = matches[0].strip()
                is_skeletal = (
                    len(method_body) < 20 or  # Very short implementation
                    "throw new UnsupportedOperationException" in method_body or
                    "TODO" in method_body or
                    "return null" in method_body or
                    "return;" in method_body or
                    not method_body  # Empty body
                )
                if is_skeletal:
                    empty_or_todo_methods += 1
        
        # Consider source skeletal if most methods are empty/minimal
        source_methods_skeletal = (empty_or_todo_methods / total_methods) > 0.5 if total_methods > 0 else False

    # Check for test framework setup
    has_test_framework = bool(test_initial.test_frameworks)
    
    # Check for testing patterns in imports
    has_testing_imports = any(
        "test" in imp.lower() or 
        "junit" in imp.lower() or 
        "assert" in imp.lower() 
        for imp in test_initial.imports
    )

    # Return true if we have strong indicators of TDD
    return (test_methods_complete and source_methods_skeletal) or (
        has_test_framework and has_testing_imports and test_methods_complete
    )

# TODO: Figure what the fuck this is
def indicates_repository_move(
    file1: FileHistory, file2: FileHistory, commit_graph: CommitGraph
) -> bool:
    """Detect if files were likely moved from another repository."""
    # Check for indicators of repository migration
    for date1, content1 in file1.content_history:
        for date2, content2 in file2.content_history:
            # Files have similar content but different creation dates
            if content1.raw_content == content2.raw_content and abs(
                date1 - date2
            ) > datetime.timedelta(days=1):
                # Check if commit is a merge or has multiple parents
                commit1 = next(
                    (commit for _, commit in file1.modifications if date1 == _), None
                )
                commit2 = next(
                    (commit for _, commit in file2.modifications if date2 == _), None
                )

                if commit1 and commit2:
                    branch_point1 = commit_graph.get_branch_point(commit1)
                    branch_point2 = commit_graph.get_branch_point(commit2)

                    # If either commit is at a branch point, likely indicates a repository merge
                    return bool(branch_point1 or branch_point2)
    return False

def analyze_tdd_patterns(matches: List[MatchedPair], commit_graph: CommitGraph) -> dict:
    """Enhanced TDD pattern analysis with more detailed reporting"""
    logging.info("Starting comprehensive TDD pattern analysis")

    results = {
        "total_matches": len(matches),
        "same_directory_matches": sum(1 for m in matches if m.directory_match),
        "test_first": 0,
        "test_after": 0,
        "same_commit_tdd": 0,
        "same_commit_unclear": 0,
        "relationship_details": [],  # New: detailed relationship info
        "relationship_types": defaultdict(int),  # Fixed: Initialize defaultdict
        "confidence_details": [],    # New: detailed confidence info
        "confidence_distribution": defaultdict(int),  # Added: Initialize defaultdict
        "multi_test_coverage": 0,
        "abstract_base_coverage": 0,
        "utility_class_coverage": 0,
        "framework_details": [],     # New: detailed framework info
        "repository_moves": 0,
        "squashed_commits": 0,
    }

    # Track source files with multiple tests
    source_to_tests = defaultdict(list)
    
    # Track frameworks across all test files
    all_frameworks = set()

    for match in matches:
        # Store detailed relationship info
        relationship_detail = {
            "test_file": match.test_file.basename,
            "source_file": match.source_file.basename,
            "type": match.relationship_type,
            "confidence": match.confidence_score,
            "directory_match": match.directory_match
        }
        results["relationship_details"].append(relationship_detail)

        # Store detailed confidence info
        confidence_detail = {
            "test_file": match.test_file.basename,
            "source_file": match.source_file.basename,
            "confidence_score": match.confidence_score,
            "factors": []  # Will store reasons for confidence score
        }
        
        # Analyze confidence factors
        if match.directory_match:
            confidence_detail["factors"].append("Same directory")
        if match.test_file.basename.replace("Test", "") == match.source_file.basename:
            confidence_detail["factors"].append("Direct name match")
        latest_test_content = match.test_file.latest_content
        if latest_test_content and match.source_file.basename[:-5] in " ".join(latest_test_content.raw_content.split()):
            confidence_detail["factors"].append("Source class referenced in test")
        
        results["confidence_details"].append(confidence_detail)

        # Track relationship types
        results["relationship_types"][match.relationship_type] += 1

        # Track confidence distribution
        confidence_bucket = round(match.confidence_score * 10) / 10
        results["confidence_distribution"][confidence_bucket] += 1

        # Add to source-test mapping
        source_to_tests[match.source_file.full_path].append(match.test_file)

        # Analyze creation pattern
        if match.test_file.creation_date < match.source_file.creation_date:
            results["test_first"] += 1
        elif match.test_file.creation_date > match.source_file.creation_date:
            results["test_after"] += 1
        else:
            if has_tdd_indicators(match.test_file, match.source_file):
                results["same_commit_tdd"] += 1
            else:
                results["same_commit_unclear"] += 1

        # Track frameworks used
        latest_test_content = match.test_file.latest_content
        if latest_test_content:
            all_frameworks.update(latest_test_content.test_frameworks)
            
            # Store detailed framework info
            framework_detail = {
                "test_file": match.test_file.basename,
                "frameworks": latest_test_content.test_frameworks,
                "test_methods": len(latest_test_content.test_methods)
            }
            results["framework_details"].append(framework_detail)

    # Calculate final metrics
    total_determinable = results["test_first"] + results["test_after"] + results["same_commit_tdd"]
    if total_determinable > 0:
        results["tdd_adoption_rate"] = (results["test_first"] + results["same_commit_tdd"]) / total_determinable

     # Convert defaultdicts to regular dicts for cleaner logging
    simple_results = {
        "total_matches": results["total_matches"],
        "same_directory_matches": results["same_directory_matches"],
        "test_first": results["test_first"],
        "test_after": results["test_after"],
        "same_commit_tdd": results["same_commit_tdd"],
        "same_commit_unclear": results["same_commit_unclear"],
        "relationship_types": dict(results["relationship_types"]),
        "confidence_distribution": dict(results["confidence_distribution"]),
        "multi_test_coverage": results["multi_test_coverage"],
        "abstract_base_coverage": results["abstract_base_coverage"],
        "utility_class_coverage": results["utility_class_coverage"],
        "repository_moves": results["repository_moves"],
        "squashed_commits": results["squashed_commits"]
    }
    if "tdd_adoption_rate" in results:
        simple_results["tdd_adoption_rate"] = results["tdd_adoption_rate"]

    logging.info("Analysis Results:")
    for key, value in simple_results.items():
        logging.info(f"{key}: {value}")

    return results

def generate_detailed_report(results: dict, output_dir: str):
    """Generate enhanced detailed analysis report in a timestamped folder"""
    # Create timestamp for the run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create a new directory for this analysis run
    run_dir = os.path.join(output_dir, f"analysis_run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    logging.info(f"Generating reports in directory: {run_dir}")

    # Create main results CSV
    main_df = pd.DataFrame([{k: v for k, v in results.items() 
                            if not isinstance(v, (list, dict, defaultdict))}])
    main_df.to_csv(os.path.join(run_dir, "tdd_analysis_summary.csv"), 
                  index=False)

    # Create detailed relationship analysis CSV
    relationship_df = pd.DataFrame(results["relationship_details"])
    relationship_df.to_csv(
        os.path.join(run_dir, "relationship_analysis.csv"),
        index=False
    )

    # Create detailed confidence analysis CSV
    confidence_df = pd.DataFrame(results["confidence_details"])
    confidence_df.to_csv(
        os.path.join(run_dir, "confidence_analysis.csv"),
        index=False
    )

    # Create detailed framework analysis CSV
    framework_df = pd.DataFrame(results["framework_details"])
    framework_df.to_csv(
        os.path.join(run_dir, "framework_analysis.csv"),
        index=False
    )

    logging.info(f"Enhanced reports generated in {run_dir}")
    
    # # Log summary statistics (For manual inspection)
    # logging.info("\nDetailed Analysis Summary:")
    # logging.info("Test Framework Usage:")
    # for framework_detail in results["framework_details"]:
    #     logging.info(f"  {framework_detail['test_file']}:")
    #     logging.info(f"    Frameworks: {', '.join(framework_detail['frameworks'])}")
    #     logging.info(f"    Test Methods: {framework_detail['test_methods']}")
    
    # logging.info("\nRelationship Analysis:")
    # for rel_detail in results["relationship_details"]:
    #     logging.info(f"  {rel_detail['test_file']} -> {rel_detail['source_file']}:")
    #     logging.info(f"    Type: {rel_detail['type']}")
    #     logging.info(f"    Confidence: {rel_detail['confidence']:.2f}")
        
    # logging.info("\nConfidence Score Breakdown:")
    # for conf_detail in results["confidence_details"]:
    #     logging.info(f"  {conf_detail['test_file']} -> {conf_detail['source_file']}:")
    #     logging.info(f"    Score: {conf_detail['confidence_score']:.2f}")
    #     logging.info(f"    Factors: {', '.join(conf_detail['factors'])}")

def main():
    logging.info("Starting enhanced TDD analysis")
    # Clone repositories
    clone_repos(REPO_URLS, CLONE_DIR)

    # Analyze each repository
    for repo_name in os.listdir(CLONE_DIR):
        repo_path = os.path.join(CLONE_DIR, repo_name)
        if os.path.isdir(repo_path):
            logging.info(f"\nAnalyzing repository: {repo_name}")

            # Get file histories and commit graph
            # TODO: Figure what is the commit graph
            file_histories, commit_graph = get_file_creation_dates(repo_path)

            # Match test and source files
            matches = match_tests_sources(file_histories, commit_graph)

            # Analyze TDD patterns
            results = analyze_tdd_patterns(matches, commit_graph)

            # Generate detailed report
            generate_detailed_report(results, OUTPUT_DIR)

    logging.info("Analysis completed")


if __name__ == "__main__":
    main()



