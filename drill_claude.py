import re
from typing import Dict, Tuple, List, NamedTuple
from dataclasses import dataclass
from datetime import datetime
import os
from collections import defaultdict
from git import Repo
from pydriller import Repository, ModificationType
import pandas as pd
from datetime import datetime


# List of fucking repository URLs
REPO_URLS = [
    "https://github.com/apache/bigtop-manager",
]

# Directory to clone repositories
CLONE_DIR = "apache_repos"

# Directory to save analysis results
OUTPUT_DIR = "results"

# Patterns to identify test files (adjust as needed)
TEST_FILE_PATTERNS = [r".*Test\.java$", r".*test_.*\.py$"]
SOURCE_FILE_PATTERNS = [r".*\.java$", r".*\.py$"]


@dataclass
class FileHistory:
    creation_date: datetime
    full_path: str
    basename: str  # Keep basename for matching
    directory: str  # Keep directory for context
    last_modified_date: datetime
    is_deleted: bool = False


class MatchedPair(NamedTuple):
    test_file: FileHistory
    source_file: FileHistory
    directory_match: bool  # Indicates if they're in the same/related directories

def clone_repos(repo_urls, clone_dir):
    if not os.path.exists(clone_dir):
        os.makedirs(clone_dir)
    for url in repo_urls:
        repo_name = url.split("/")[-1].replace(".git", "")
        repo_path = os.path.join(clone_dir, repo_name)
        if not os.path.exists(repo_path):
            print(f"Cloning {repo_name}...")
            Repo.clone_from(url, repo_path)
        else:
            print(f"{repo_name} already cloned.")

def get_file_creation_dates(repo_path: str) -> Dict[str, FileHistory]:
    """
    Track file creation dates while maintaining both full paths and basenames for matching.
    """
    file_histories: Dict[str, FileHistory] = {}
    basename_map: Dict[str, List[str]] = defaultdict(
        list
    )  # Maps basenames to full paths

    repo = Repo(repo_path)
    default_branch = repo.active_branch.name

    for commit in Repository(
        repo_path, only_in_branch=default_branch
    ).traverse_commits():
        for modification in commit.modified_files:
            current_path = modification.new_path or modification.old_path
            if not current_path:
                continue

            normalized_path = os.path.normpath(current_path)
            basename = os.path.basename(normalized_path)
            directory = os.path.dirname(normalized_path)

            if modification.change_type == ModificationType.ADD:
                if normalized_path not in file_histories:
                    file_histories[normalized_path] = FileHistory(
                        creation_date=commit.author_date,
                        full_path=normalized_path,
                        basename=basename,
                        directory=directory,
                        last_modified_date=commit.author_date,
                        is_deleted=False,
                    )
                    basename_map[basename].append(normalized_path)

            elif modification.change_type == ModificationType.DELETE:
                if normalized_path in file_histories:
                    file_histories[normalized_path].is_deleted = True

            elif modification.change_type == ModificationType.MODIFY:
                if normalized_path in file_histories:
                    file_histories[normalized_path].last_modified_date = (
                        commit.author_date
                    )

    return file_histories, basename_map


def match_tests_sources(file_histories: Dict[str, FileHistory], 
                       basename_map: Dict[str, List[str]]) -> List[MatchedPair]:
    matches = []
    test_pattern = re.compile(r'(.+)Test\.java$')
    
    test_files = {
        path: history 
        for path, history in file_histories.items() 
        if test_pattern.match(history.basename) and not history.is_deleted
    }
    
    for test_path, test_history in test_files.items():
        match = test_pattern.match(test_history.basename)
        if not match:
            continue
            
        source_basename = f"{match.group(1)}.java"
        potential_source_paths = basename_map.get(source_basename, [])
        
        if not potential_source_paths:
            continue
            
        if len(potential_source_paths) == 1:
            source_path = potential_source_paths[0]
            if source_path in file_histories and not file_histories[source_path].is_deleted:
                matches.append(MatchedPair(
                    test_file=test_history,
                    source_file=file_histories[source_path],
                    directory_match=test_history.directory == file_histories[source_path].directory
                ))
        else:
            best_match = None
            for source_path in potential_source_paths:
                if source_path not in file_histories or file_histories[source_path].is_deleted:
                    continue
                    
                source_history = file_histories[source_path]
                
                if test_history.directory == source_history.directory:
                    best_match = source_history
                    break
                    
                if is_related_directory(test_history.directory, source_history.directory):
                    best_match = source_history
            
            if best_match:
                matches.append(MatchedPair(
                    test_file=test_history,
                    source_file=best_match,
                    directory_match=test_history.directory == best_match.directory
                ))
    
    return matches  # Added return statement


def is_related_directory(test_dir: str, source_dir: str) -> bool:
    """
    Check if test and source directories are related (e.g., src/test vs src/main).
    """
    # Split paths into components
    test_components = test_dir.split(os.sep)
    source_components = source_dir.split(os.sep)

    # Common patterns for related directories
    patterns = [("test", "main"), ("tests", "src"), ("test", "src"), ("tests", "main")]

    # Check if directories differ only by these patterns
    for test_pattern, source_pattern in patterns:
        test_normalized = [
            c if c != test_pattern else source_pattern for c in test_components
        ]
        if test_normalized == source_components:
            return True

    return False


def analyze_tdd_patterns(matches: List[MatchedPair]) -> dict:
    """
    Analyze TDD patterns based on matched pairs.
    """
    results = {
        "total_matches": len(matches),
        "same_directory_matches": sum(1 for m in matches if m.directory_match),
        "test_first": 0,
        "test_after": 0,
        "same_commit": 0,
    }

    for match in matches:
        test_date = match.test_file.creation_date
        source_date = match.source_file.creation_date

        if test_date < source_date:
            results["test_first"] += 1
        elif test_date > source_date:
            results["test_after"] += 1
        else:
            results["same_commit"] += 1

    return results


def analyze_repos(clone_dir):
    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(OUTPUT_DIR, f"claude_tdd_results_{timestamp}.csv")

    for repo_name in os.listdir(clone_dir):
        repo_path = os.path.join(clone_dir, repo_name)
        if os.path.isdir(repo_path):
            print(f"Analyzing {repo_name}...")

            # Get file histories and basename mapping
            file_histories, basename_map = get_file_creation_dates(repo_path)

            # print(file_histories, basename_map)

            # Match test and source files
            matches = match_tests_sources(file_histories, basename_map)

            # print(matches)

            # Analyze TDD patterns
            tdd_analysis = analyze_tdd_patterns(matches)

            results.append(
                {
                    "repository": repo_name,
                    "total_matches": tdd_analysis["total_matches"],
                    "same_directory_matches": tdd_analysis["same_directory_matches"],
                    "test_first_count": tdd_analysis["test_first"],
                    "test_after_count": tdd_analysis["test_after"],
                    "same_commit_count": tdd_analysis["same_commit"],
                    "tdd_ratio": (
                        tdd_analysis["test_first"] / tdd_analysis["total_matches"]
                        if tdd_analysis["total_matches"] > 0
                        else 0
                    ),
                }
            )

    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"Analysis complete. Results saved to {output_file}")

def main():
    clone_repos(REPO_URLS, CLONE_DIR)
    analyze_repos(CLONE_DIR)


if __name__ == "__main__":
    main()
