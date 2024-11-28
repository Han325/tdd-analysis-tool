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
if not os.path.exists('logs'):
    os.makedirs('logs')

# Create timestamped log filename
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = os.path.join('logs', f'tdd_analysis_{timestamp}.log')

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
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
OUTPUT_DIR = "results"

# Enhanced patterns for test detection
TEST_PATTERNS = {
    "java": {
        "file_patterns": [
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
    },
    "python": {
        "file_patterns": [r"test_.*\.py$", r".*_test\.py$", r".*_tests\.py$"],
        "frameworks": ["unittest", "pytest", "nose", "doctest"],
        "decorators": ["@pytest.fixture", "@pytest.mark.parametrize"],
    },
}


@dataclass
class FileContent:
    raw_content: str
    imports: List[str]
    class_names: List[str]
    method_names: List[str]
    is_abstract: bool
    is_utility: bool
    test_frameworks: List[str]
    test_methods: List[str]
    dependencies: List[str]


@dataclass
class FileHistory:
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
    is_test: bool = False
    is_abstract: bool = False
    is_utility: bool = False


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
    logging.info(f"Starting repository cloning process for {len(repo_urls)} repos")
    if not os.path.exists(clone_dir):
        os.makedirs(clone_dir)
        logging.debug(f"Created clone directory: {clone_dir}")

    for url in repo_urls:
        repo_name = url.split("/")[-1].replace(".git", "")
        repo_path = os.path.join(clone_dir, repo_name)
        logging.debug(f"Processing repository: {repo_name}")

        if not os.path.exists(repo_path):
            logging.info(f"Cloning {repo_name}...")
            try:
                Repo.clone_from(url, repo_path)
                logging.info(f"Successfully cloned {repo_name}")
            except Exception as e:
                logging.error(f"Failed to clone {repo_name}: {str(e)}")
        else:
            logging.info(f"{repo_name} already exists at {repo_path}")


def analyze_java_content(content: str) -> FileContent:
    try:
        tree = javalang.parse.parse(content)
        imports = [imp.path for imp in tree.imports]

        class_names = []
        method_names = []
        is_abstract = False
        is_utility = False
        test_frameworks = []
        test_methods = []
        dependencies = []

        for path, node in tree.filter(ClassDeclaration):
            class_names.append(node.name)

            # Check for abstract class
            if "abstract" in node.modifiers:
                is_abstract = True

            # Check for utility class (all static methods)
            if all("static" in method.modifiers for method in node.methods):
                is_utility = True

            # Analyze methods
            for method in node.methods:
                method_names.append(method.name)

                # Check for test methods
                for annotation in method.annotations:
                    if annotation.name in TEST_PATTERNS["java"]["annotations"]:
                        test_methods.append(method.name)

        # Check for test framework imports
        for framework in TEST_PATTERNS["java"]["frameworks"]:
            if any(framework in imp for imp in imports):
                test_frameworks.append(framework)

        # Extract dependencies
        dependencies = imports + [
            ref.name for ref in tree.filter(javalang.tree.ReferenceType)
        ]

        return FileContent(
            raw_content=content,
            imports=imports,
            class_names=class_names,
            method_names=method_names,
            is_abstract=is_abstract,
            is_utility=is_utility,
            test_frameworks=test_frameworks,
            test_methods=test_methods,
            dependencies=dependencies,
        )
    except:
        logging.error(f"Failed to parse Java content")
        return FileContent("", [], [], [], False, False, [], [], [])


def analyze_python_content(content: str) -> FileContent:
    try:
        tree = ast.parse(content)
        imports = []
        class_names = []
        method_names = []
        is_abstract = False
        is_utility = True  # Assume utility until we find instance methods
        test_frameworks = []
        test_methods = []
        dependencies = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(name.name for name in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(f"{node.module}.{name.name}" for name in node.names)
            elif isinstance(node, ast.ClassDef):
                class_names.append(node.name)

                # Check for abstract base classes
                if any(
                    base.id == "ABC"
                    for base in node.bases
                    if isinstance(base, ast.Name)
                ):
                    is_abstract = True

                # Check methods
                for child in node.body:
                    if isinstance(child, ast.FunctionDef):
                        method_names.append(child.name)

                        # Check for instance methods (non-static)
                        if child.args.args and child.args.args[0].arg == "self":
                            is_utility = False

                        # Check for test methods
                        if child.name.startswith("test_"):
                            test_methods.append(child.name)

                        # Check for test decorators
                        for decorator in child.decorator_list:
                            if isinstance(decorator, ast.Name):
                                decorator_name = f"@{decorator.id}"
                                if (
                                    decorator_name
                                    in TEST_PATTERNS["python"]["decorators"]
                                ):
                                    test_methods.append(child.name)

        # Check for test framework imports
        for framework in TEST_PATTERNS["python"]["frameworks"]:
            if framework in imports:
                test_frameworks.append(framework)

        return FileContent(
            raw_content=content,
            imports=imports,
            class_names=class_names,
            method_names=method_names,
            is_abstract=is_abstract,
            is_utility=is_utility,
            test_frameworks=test_frameworks,
            test_methods=test_methods,
            dependencies=imports,
        )
    except:
        logging.error(f"Failed to parse Python content")
        return FileContent("", [], [], [], False, False, [], [], [])


def get_file_creation_dates(
    repo_path: str,
) -> Tuple[Dict[str, FileHistory], CommitGraph]:
    logging.info(f"Starting enhanced file history analysis for repo: {repo_path}")
    file_histories: Dict[str, FileHistory] = {}
    commit_graph = CommitGraph()

    try:
        repo = Repo(repo_path)
        default_branch = repo.active_branch.name

        # First pass: Build commit graph
        for commit in Repository(repo_path).traverse_commits():
            commit_graph.add_commit(
                commit.hash, [p.hash for p in commit.parents], commit.branches
            )

        # Second pass: Analyze files
        for commit in Repository(repo_path).traverse_commits():
            for modification in commit.modified_files:
                current_path = modification.new_path or modification.old_path
                if not current_path:
                    continue

                normalized_path = os.path.normpath(current_path)

                # Initialize or update file history
                if normalized_path not in file_histories:
                    file_histories[normalized_path] = FileHistory(
                        creation_date=commit.author_date,
                        full_path=normalized_path,
                        basename=os.path.basename(normalized_path),
                        directory=os.path.dirname(normalized_path),
                        last_modified_date=commit.author_date,
                        creation_commit=commit.hash,
                        modifications=[(commit.author_date, commit.hash)],
                        content_history=[],
                        related_files=set(),
                    )
                else:
                    file_histories[normalized_path].modifications.append(
                        (commit.author_date, commit.hash)
                    )
                    file_histories[normalized_path].last_modified_date = (
                        commit.author_date
                    )

                # Analyze file content
                if modification.source_code:
                    if current_path.endswith(".java"):
                        content = analyze_java_content(modification.source_code)
                    elif current_path.endswith(".py"):
                        content = analyze_python_content(modification.source_code)
                    else:
                        continue

                    file_histories[normalized_path].content_history.append(
                        (commit.author_date, content)
                    )

                    # Update file attributes
                    file_histories[normalized_path].is_abstract = content.is_abstract
                    file_histories[normalized_path].is_utility = content.is_utility
                    file_histories[normalized_path].is_test = bool(content.test_methods)

                    # Track related files through imports and dependencies
                    for dep in content.dependencies:
                        related_path = os.path.join(
                            os.path.dirname(normalized_path),
                            f"{dep.replace('.', '/')}.java",
                        )
                        file_histories[normalized_path].related_files.add(related_path)

                if modification.change_type == ModificationType.DELETE:
                    file_histories[normalized_path].is_deleted = True

        return file_histories, commit_graph

    except Exception as e:
        logging.error(f"Error processing repository: {str(e)}")
        raise


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
    logging.debug(
        f"Analyzing directory relationship between:\nTest: {test_dir}\nSource: {source_dir}"
    )

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
        # Python patterns
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

    # Base confidence from naming convention
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


def match_tests_sources(
    file_histories: Dict[str, FileHistory], commit_graph: CommitGraph
) -> List[MatchedPair]:
    logging.info("Starting enhanced test-source file matching process")
    matches = []

    # Group files by their base names (without Test/Tests suffix)
    basename_groups = defaultdict(list)
    for path, history in file_histories.items():
        if not history.is_deleted:
            base = re.sub(r"Test[s]?\.java$", ".java", history.basename)
            basename_groups[base].append(history)

    # Process each group
    for base, histories in basename_groups.items():
        test_files = [h for h in histories if h.is_test]
        source_files = [h for h in histories if not h.is_test]

        for test_file in test_files:
            best_match = None
            best_confidence = 0.0
            relationship_type = "direct"

            # Check each potential source file
            for source_file in source_files:
                confidence = calculate_match_confidence(test_file, source_file)

                # Adjust confidence based on commit history
                if test_file.creation_commit == source_file.creation_commit:
                    # Check for same-commit TDD patterns
                    if has_tdd_indicators(test_file, source_file):
                        confidence += 0.1

                # Check if files were moved from another repository
                if indicates_repository_move(test_file, source_file, commit_graph):
                    confidence += 0.05

                if confidence > best_confidence:
                    best_confidence = confidence
                    best_match = source_file

                    # Determine relationship type
                    if source_file.is_abstract:
                        relationship_type = "abstract"
                    elif source_file.is_utility:
                        relationship_type = "utility"

            if best_match and best_confidence > 0.3:  # Confidence threshold
                matches.append(
                    MatchedPair(
                        test_file=test_file,
                        source_file=best_match,
                        directory_match=test_file.directory == best_match.directory,
                        confidence_score=best_confidence,
                        relationship_type=relationship_type,
                    )
                )

    return matches


def has_tdd_indicators(test_file: FileHistory, source_file: FileHistory) -> bool:
    """Check for indicators of TDD when files are committed together."""
    # Get content at creation time
    test_initial = next(
        (c for d, c in test_file.content_history if d == test_file.creation_date), None
    )
    source_initial = next(
        (c for d, c in source_file.content_history if d == source_file.creation_date),
        None,
    )

    if not (test_initial and source_initial):
        return False

    # Indicators of TDD:
    # 1. Test file has complete test methods while source is skeletal
    test_completeness = len(test_initial.test_methods) > 0 and all(
        len(m) > 10 for m in test_initial.test_methods
    )  # Non-empty test methods

    source_skeletal = len(source_initial.method_names) > 0 and all(
        len(m) < 5 for m in source_initial.method_names
    )  # Minimal implementations

    # 2. Test has assertions/expectations defined
    has_assertions = any(
        pattern in test_initial.raw_content
        for pattern in ["assert", "expect", "should", "verify"]
    )

    # 3. Test frameworks/annotations present in initial commit
    has_test_framework = bool(test_initial.test_frameworks)

    return (test_completeness and source_skeletal) or (
        has_assertions and has_test_framework
    )


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
    """Enhanced TDD pattern analysis with detailed categorization."""
    logging.info("Starting comprehensive TDD pattern analysis")

    results = {
        "total_matches": len(matches),
        "same_directory_matches": sum(1 for m in matches if m.directory_match),
        "test_first": 0,
        "test_after": 0,
        "same_commit_tdd": 0,
        "same_commit_unclear": 0,
        "relationship_types": defaultdict(int),
        "confidence_distribution": defaultdict(int),
        "multi_test_coverage": 0,  # Cases where multiple tests cover one source
        "abstract_base_coverage": 0,  # Tests covering abstract classes
        "utility_class_coverage": 0,  # Tests covering utility classes
        "framework_distribution": defaultdict(int),
        "repository_moves": 0,
        "squashed_commits": 0,
    }

    # Track source files with multiple tests
    source_to_tests = defaultdict(list)

    for match in matches:
        # Track relationship types
        results["relationship_types"][match.relationship_type] += 1

        # Track confidence distribution
        confidence_bucket = (
            round(match.confidence_score * 10) / 10
        )  # Round to nearest 0.1
        results["confidence_distribution"][confidence_bucket] += 1

        # Add to source-test mapping
        source_to_tests[match.source_file.full_path].append(match.test_file)

        # Analyze creation pattern
        if match.test_file.creation_date < match.source_file.creation_date:
            results["test_first"] += 1
        elif match.test_file.creation_date > match.source_file.creation_date:
            results["test_after"] += 1
        else:
            # Same commit analysis
            if has_tdd_indicators(match.test_file, match.source_file):
                results["same_commit_tdd"] += 1
            else:
                results["same_commit_unclear"] += 1

        # Check for special cases
        if match.source_file.is_abstract:
            results["abstract_base_coverage"] += 1
        if match.source_file.is_utility:
            results["utility_class_coverage"] += 1

        # Check for repository moves
        if indicates_repository_move(match.test_file, match.source_file, commit_graph):
            results["repository_moves"] += 1

        # Track test framework usage
        latest_test_content = (
            match.test_file.content_history[-1][1]
            if match.test_file.content_history
            else None
        )
        if latest_test_content:
            for framework in latest_test_content.test_frameworks:
                results["framework_distribution"][framework] += 1

    # Analyze multiple test coverage
    for source_path, test_files in source_to_tests.items():
        if len(test_files) > 1:
            results["multi_test_coverage"] += 1

    # Calculate TDD adoption metrics
    total_determinable = (
        results["test_first"] + results["test_after"] + results["same_commit_tdd"]
    )

    if total_determinable > 0:
        results["tdd_adoption_rate"] = (
            results["test_first"] + results["same_commit_tdd"]
        ) / total_determinable

    logging.info("Analysis Results:")
    for key, value in results.items():
        logging.info(f"{key}: {value}")

    return results


def generate_detailed_report(results: dict, output_dir: str):
    """Generate a detailed analysis report with visualizations."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create detailed CSV reports
    main_df = pd.DataFrame([results])
    main_df.to_csv(
        os.path.join(output_dir, f"tdd_analysis_{timestamp}.csv"), index=False
    )

    # Create relationship type distribution
    pd.DataFrame(
        results["relationship_types"].items(), columns=["type", "count"]
    ).to_csv(
        os.path.join(output_dir, f"relationship_types_{timestamp}.csv"), index=False
    )

    # Create framework distribution
    pd.DataFrame(
        results["framework_distribution"].items(), columns=["framework", "count"]
    ).to_csv(
        os.path.join(output_dir, f"framework_distribution_{timestamp}.csv"), index=False
    )

    # Create confidence distribution
    pd.DataFrame(
        results["confidence_distribution"].items(), columns=["confidence", "count"]
    ).to_csv(
        os.path.join(output_dir, f"confidence_distribution_{timestamp}.csv"),
        index=False,
    )

    logging.info(f"Detailed reports generated in {output_dir}")


def main():
    logging.info("Starting enhanced TDD analysis")
    try:
        # Clone repositories
        clone_repos(REPO_URLS, CLONE_DIR)

        # Analyze each repository
        for repo_name in os.listdir(CLONE_DIR):
            repo_path = os.path.join(CLONE_DIR, repo_name)
            if os.path.isdir(repo_path):
                logging.info(f"\nAnalyzing repository: {repo_name}")

                # Get file histories and commit graph
                file_histories, commit_graph = get_file_creation_dates(repo_path)

                # Match test and source files
                matches = match_tests_sources(file_histories, commit_graph)

                # Analyze TDD patterns
                results = analyze_tdd_patterns(matches, commit_graph)

                # Generate detailed report
                generate_detailed_report(results, OUTPUT_DIR)

    except Exception as e:
        logging.error(f"Fatal error in main execution: {str(e)}")

    logging.info("Analysis completed")


if __name__ == "__main__":
    main()
