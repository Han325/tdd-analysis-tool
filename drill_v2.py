import re
from typing import Dict, Tuple, List, NamedTuple, Set, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import os
from collections import defaultdict
from git import Repo, Commit
from pydriller import Repository, ModificationType
import pandas as pd
import logging
import networkx as nx

# Create logs directory if it doesn't exist
if not os.path.exists("logs"):
    os.makedirs("logs")

# Create timestamped log filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join("logs", f"tdd_analysis_{timestamp}.log")

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
)

# List of repository URLs (Replace with your own urls)
REPO_URLS = [
    # "https://github.com/apache/bigtop-manager",
    # "https://github.com/apache/commons-csv",
    "https://github.com/apache/doris-kafka-connector",
    # "https://github.com/apache/struts-intellij-plugin",
    # "https://github.com/apache/shiro",
    # "https://github.com/apache/hbase",
    # "https://github.com/apache/doris-manager",
    # "https://github.com/apache/commons-io",
    # "https://github.com/apache/tomcat"
]

# Directory to clone repositories
CLONE_DIR = "apache_repos"

# Directory to save analysis results
OUTPUT_DIR = "results"

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


@dataclass
class CommitAnalysis:
    """Enhanced commit analysis for TDD patterns"""
    hash: str
    message: str
    date: datetime
    modified_files: List[Tuple[str, FileContent]]  # [(path, content)]
    lines_added: int
    lines_removed: int
    is_merge: bool

    @property
    def size_category(self) -> str:
        """Categorize commit size based on changes"""
        total_changes = self.lines_added + self.lines_removed
        num_files = len(self.modified_files)
        if num_files <= 2 and total_changes <= 50:
            return "small"
        elif num_files <= 5 and total_changes <= 200:
            return "medium"
        else:
            return "large"
    
    def has_tdd_message_indicators(self) -> bool:
        """Analyzes commit messages for realistic TDD patterns."""
        msg_lower = self.message.lower()
        
        patterns = [
            r"\btdd\b",
            r"test[s]?\s*[:+]\s*\w+",
            r"add(ed|ing)?\s+tests?",
            r"\btests?\s+first\b", 
            r"test[s]?\s+[+&]\s+impl",
        ]
        
        return any(re.search(pattern, msg_lower) for pattern in patterns)

    def analyze_commit_context(self) -> bool:
        """Combines message analysis with code change context"""
        has_test_indicator = self.has_tdd_message_indicators()
        test_files_added = any('test' in path.lower() for path, _ in self.modified_files)
        
        # Check for test-related content in modified files
        test_content_changes = False
        for _, content in self.modified_files:
            if content.test_methods or content.test_frameworks:
                test_content_changes = True
                break
                
        return has_test_indicator and (test_files_added or test_content_changes)

    @property
    def commit_type(self) -> str:
        """Categorize commit based on changes"""
        if self.is_merge:
            return "merge"
        if self.analyze_commit_context():
            return "test_related"
        return "implementation"


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
        import logging
        py4j_logger = logging.getLogger("py4j")
        py4j_logger.setLevel(logging.WARNING)  # Only show warnings and errors
        
        # Connect to the Java Gateway
        from py4j.java_gateway import JavaGateway, GatewayParameters
        logging.info("Connecting to Java Gateway")
        
        gateway = JavaGateway(
            gateway_parameters=GatewayParameters(
                auto_convert=True,
                auto_field=True
            )
        )
        
        parser = gateway.entry_point
        result = parser.parseJavaContent(content)
        
        if result is None:
            logging.error("Received null result from Java parser")
            return create_empty_file_content(content)
            
        # Get basic information
        imports = list(result.getImports())
        class_names = list(result.getClasses())
        method_names = list(result.getMethods())
        test_methods = list(result.getTestMethods())
        
        # Clean logging of what we found
        logging.info(f"Parsed Java file with {len(class_names)} classes")
        logging.info(f"Found {len(method_names)} methods, {len(test_methods)} test methods")
        logging.debug(f"Imports: {', '.join(imports)}")
        logging.debug(f"Classes: {', '.join(class_names)}")
        logging.debug(f"Methods: {', '.join(method_names)}")
        if test_methods:
            logging.debug(f"Test methods: {', '.join(test_methods)}")
        
        # Get test frameworks from imports
        test_frameworks = []
        for framework in TEST_PATTERNS["java"]["frameworks"]:
            if any(framework in imp for imp in imports):
                test_frameworks.append(framework)
                logging.debug(f"Found test framework: {framework}")
        
        return FileContent(
            raw_content=content,
            imports=imports,
            class_names=class_names,
            method_names=method_names,
            is_abstract=result.isAbstract(),
            is_utility=result.isUtility(),
            test_frameworks=test_frameworks,
            test_methods=test_methods,
            dependencies=list(set(imports))
        )
        
    except Exception as e:
        logging.error(f"Error in analyze_java_content: {str(e)}", exc_info=True)
        return create_empty_file_content(content)

def create_empty_file_content(content: str) -> FileContent:
    """Helper function to create an empty FileContent object"""
    return FileContent(
        raw_content=content,
        imports=[],
        class_names=[],
        method_names=[],
        is_abstract=False,
        is_utility=False,
        test_frameworks=[],
        test_methods=[],
        dependencies=[]
    )

def get_file_creation_dates(
    repo_path: str,
) -> Tuple[Dict[str, FileHistory], CommitGraph]:
    logging.info(f"Starting file history analysis for repo: {repo_path}")
    file_histories: Dict[str, FileHistory] = {}
    commit_graph = CommitGraph()
    commit_analyses = []  # New list to store commit analyses


    try:
        repo = Repo(repo_path)
        default_branch = repo.active_branch.name
        logging.debug(f"Using default branch: {default_branch}")

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

            logging.debug(
                f"Processing commit {commit.hash[:8]} from {commit.author_date}"
            )

            modified_files = []
            total_lines_added = 0
            total_lines_removed = 0

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

                # Track lines changed
                total_lines_added += modification.added_lines
                total_lines_removed += modification.deleted_lines

                if modification.change_type == ModificationType.ADD:
                    if normalized_path not in file_histories:
                        # Parse content based on file type
                        content = FileContent("", [], [], [], False, False, [], [], [])
                        if normalized_path.endswith(".java"):
                            content = analyze_java_content(modification.source_code)
                            modified_files.append((normalized_path, content))

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
                            modified_files.append((normalized_path, content))

                        file_histories[normalized_path].content_history.append(
                            (commit.author_date, content)
                        )

            # Create commit analysis after processing all modifications
            commit_analyses.append(CommitAnalysis(
                    hash=commit.hash,
                    message=commit.msg,
                    date=commit.author_date,
                    modified_files=modified_files,
                    lines_added=sum(m.added_lines for m in commit.modified_files),
                    lines_removed=sum(m.deleted_lines for m in commit.modified_files),
                    is_merge=len(commit.parents) > 1
                ))
                    
            logging.info(f"Processed {commit_count} commits")
            logging.info(f"Found {len(file_histories)} unique files")

    except Exception as e:
        logging.error(f"Error processing repository: {str(e)}")
        raise

    return file_histories, commit_graph, commit_analyses


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


def match_tests_sources(
    file_histories: Dict[str, FileHistory], commit_graph: CommitGraph
) -> List[MatchedPair]:
    logging.info("Starting enhanced test-source file matching process")
    matches = []

    # Group files by their base names (without Test/Tests suffix)
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
    logging.info(f"Number of base name groups: {len(basename_groups)}")

    # Process each group
    for base, histories in basename_groups.items():
        logging.debug(f"\nProcessing base name group: {base}")

        test_files = [h for h in histories if h.is_test]
        source_files = [h for h in histories if not h.is_test]

        logging.debug(
            f"Found {len(test_files)} test files and {len(source_files)} source files in group"
        )

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
                        logging.debug(
                            f"Added TDD indicator bonus. New confidence: {confidence}"
                        )

                # Check if files were moved from another repository
                if indicates_repository_move(test_file, source_file, commit_graph):
                    confidence += 0.05
                    logging.debug(
                        f"Added repository move bonus. New confidence: {confidence}"
                    )

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

    logging.info(f"Total matches found: {len(matches)}")
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
            ) > timedelta(days=1):
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


def analyze_tdd_patterns(matches: List[MatchedPair], commit_graph: CommitGraph, commit_analyses: List[CommitAnalysis]) -> dict:
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

         # Added new metrics
        "commit_sizes": {
            "small": {"test_first": 0, "test_after": 0, "same_commit": 0},
            "medium": {"test_first": 0, "test_after": 0, "same_commit": 0},
            "large": {"test_first": 0, "test_after": 0, "same_commit": 0}
        },
        "commit_messages": {
            "tdd_indicated": 0,
            "test_focused": 0
        },
        "commit_patterns": {
            "test_related": 0,
            "implementation": 0,
            "merge": 0
        },
        "context_analysis": {
            "message_and_content_match": 0,
            "message_only": 0,
            "content_only": 0
        }
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

        # # Find relevant commits
        test_commit = next((ca for ca in commit_analyses 
                          if any(path == match.test_file.full_path 
                                for path, _ in ca.modified_files)), None)
        source_commit = next((ca for ca in commit_analyses 
                            if any(path == match.source_file.full_path 
                                  for path, _ in ca.modified_files)), None)

        if test_commit and source_commit:
            size_category = test_commit.size_category
            
            # Analyze creation pattern (combining original and new logic)
            if match.test_file.creation_date < match.source_file.creation_date:
                results["test_first"] += 1
                results["commit_sizes"][size_category]["test_first"] += 1
            elif match.test_file.creation_date > match.source_file.creation_date:
                results["test_after"] += 1
                results["commit_sizes"][size_category]["test_after"] += 1
            else:
                if has_tdd_indicators(match.test_file, match.source_file):
                    results["same_commit_tdd"] += 1
                    results["commit_sizes"][size_category]["same_commit"] += 1
                else:
                    results["same_commit_unclear"] += 1
            
            # Enhanced commit analysis
            results["commit_patterns"][test_commit.commit_type] += 1
            if test_commit.has_tdd_message_indicators():
                results["commit_messages"]["tdd_indicated"] += 1
                if test_commit.analyze_commit_context():
                    results["context_analysis"]["message_and_content_match"] += 1
                else:
                    results["context_analysis"]["message_only"] += 1
            elif test_commit.analyze_commit_context():
                results["context_analysis"]["content_only"] += 1

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
        # Commented out following output to focus core data related to RQ
        # "relationship_types": dict(results["relationship_types"]),
        # "confidence_distribution": dict(results["confidence_distribution"]),
        # "multi_test_coverage": results["multi_test_coverage"],
        # "abstract_base_coverage": results["abstract_base_coverage"],
        # "utility_class_coverage": results["utility_class_coverage"],
        # "repository_moves": results["repository_moves"],
        # "squashed_commits": results["squashed_commits"],

        # Add commit size metrics
        "commit_sizes": {
            "small": dict(results["commit_sizes"]["small"]),
            "medium": dict(results["commit_sizes"]["medium"]),
            "large": dict(results["commit_sizes"]["large"])
        },
        # Add commit message metrics
        "commit_messages": dict(results["commit_messages"]),
         # Add commit pattern metrics
        "commit_patterns": dict(results["commit_patterns"]),
        "context_analysis": dict(results["context_analysis"])
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
    main_summary = {
        "total_matches": results["total_matches"],
        "same_directory_matches": results["same_directory_matches"],
        "test_first": results["test_first"],
        "test_after": results["test_after"],
        "same_commit_tdd": results["same_commit_tdd"],
        "same_commit_unclear": results["same_commit_unclear"],
        "tdd_adoption_rate": results.get("tdd_adoption_rate", 0)
    }
    main_df = pd.DataFrame([main_summary])
    main_df.to_csv(os.path.join(run_dir, "tdd_analysis_summary.csv"), 
                  index=False)

    # Commented out following csv output to focus core data related to RQ
    # # Create detailed relationship analysis CSV
    # relationship_df = pd.DataFrame(results["relationship_details"])
    # relationship_df.to_csv(
    #     os.path.join(run_dir, "relationship_analysis.csv"),
    #     index=False
    # )

    # # Create detailed confidence analysis CSV
    # confidence_df = pd.DataFrame(results["confidence_details"])
    # confidence_df.to_csv(
    #     os.path.join(run_dir, "confidence_analysis.csv"),
    #     index=False
    # )

    # # Create detailed framework analysis CSV
    # framework_df = pd.DataFrame(results["framework_details"])
    # framework_df.to_csv(
    #     os.path.join(run_dir, "framework_analysis.csv"),
    #     index=False
    # )

    size_data = []
    for size, patterns in results["commit_sizes"].items():
        row = {"size_category": size}
        row.update(patterns)
        size_data.append(row)
    size_df = pd.DataFrame(size_data)
    size_df.to_csv(
        os.path.join(run_dir, "commit_size_analysis.csv"),
        index=False
    )

    message_df = pd.DataFrame([results["commit_messages"]])
    message_df.to_csv(
        os.path.join(run_dir, "commit_message_patterns.csv"),
        index=False
    )

    commit_patterns_df = pd.DataFrame([results["commit_patterns"]])
    commit_patterns_df.to_csv(os.path.join(run_dir, "commit_patterns.csv"), index=False)

    context_analysis_df = pd.DataFrame([results["context_analysis"]])
    context_analysis_df.to_csv(os.path.join(run_dir, "context_analysis.csv"), index=False)

    combined_analysis = {
        **main_summary,
        "small_test_first": results["commit_sizes"]["small"]["test_first"],
        "small_test_after": results["commit_sizes"]["small"]["test_after"],
        "small_same_commit": results["commit_sizes"]["small"]["same_commit"],
        "medium_test_first": results["commit_sizes"]["medium"]["test_first"],
        "medium_test_after": results["commit_sizes"]["medium"]["test_after"],
        "medium_same_commit": results["commit_sizes"]["medium"]["same_commit"],
        "large_test_first": results["commit_sizes"]["large"]["test_first"],
        "large_test_after": results["commit_sizes"]["large"]["test_after"],
        "large_same_commit": results["commit_sizes"]["large"]["same_commit"],
        "tdd_indicated_commits": results["commit_messages"]["tdd_indicated"],
        "test_focused_commits": results["commit_messages"]["test_focused"],
        "test_related_commits": results["commit_patterns"]["test_related"],
        "implementation_commits": results["commit_patterns"]["implementation"],
        "merge_commits": results["commit_patterns"]["merge"],
        "message_content_match": results["context_analysis"]["message_and_content_match"],
        "message_only_indicators": results["context_analysis"]["message_only"],
        "content_only_indicators": results["context_analysis"]["content_only"]
    }
    combined_df = pd.DataFrame([combined_analysis])
    combined_df.to_csv(
        os.path.join(run_dir, "combined_analysis.csv"),
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
    try:
        # Clone repositories
        clone_repos(REPO_URLS, CLONE_DIR)

        # Analyze each repository
        for repo_url in REPO_URLS:
            repo_name = repo_url.split("/")[-1].replace(".git", "")
            repo_path = os.path.join(CLONE_DIR, repo_name)

            if not os.path.exists(repo_path):
                logging.error(f"Repository {repo_name} does not exist at {repo_path}. Skipping...")
                continue
            if os.path.isdir(repo_path):
                logging.info(f"\nAnalyzing repository: {repo_name}")

                # Get file histories and commit graph
                file_histories, commit_graph, commit_analyses = get_file_creation_dates(repo_path)

                # Match test and source files
                matches = match_tests_sources(file_histories, commit_graph)

                # Analyze TDD patterns
                results = analyze_tdd_patterns(matches, commit_graph, commit_analyses)

                # Generate detailed report
                generate_detailed_report(results, OUTPUT_DIR)

    except Exception as e:
        logging.error(f"Fatal error in main execution: {str(e)}")

    logging.info("Analysis completed")


if __name__ == "__main__":
    main()
