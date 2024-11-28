import re
import os
import pandas as pd
import logging
from typing import Dict, Tuple, List, NamedTuple
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict
from git import Repo
from pydriller import Repository, ModificationType


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

# Patterns to identify test files (adjust as needed)
TEST_FILE_PATTERNS = [r".*Test\.java$", r".*test_.*\.py$"]
SOURCE_FILE_PATTERNS = [r".*\.java$", r".*\.py$"]


@dataclass
class FileHistory:
    creation_date: datetime
    full_path: str
    basename: str  
    directory: str  
    last_modified_date: datetime
    is_deleted: bool = False


class MatchedPair(NamedTuple):
    test_file: FileHistory
    source_file: FileHistory
    directory_match: bool  

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

def get_file_creation_dates(repo_path: str) -> Dict[str, FileHistory]:
    logging.info(f"Starting file history analysis for repo: {repo_path}")
    file_histories: Dict[str, FileHistory] = {}
    basename_map: Dict[str, List[str]] = defaultdict(list)

    try:
        repo = Repo(repo_path)
        default_branch = repo.active_branch.name
        logging.debug(f"Using default branch: {default_branch}")

        commit_count = 0
        for commit in Repository(repo_path, only_in_branch=default_branch).traverse_commits():
            commit_count += 1
            logging.debug(f"Processing commit {commit.hash[:8]} from {commit.author_date}")
            
            for modification in commit.modified_files:
                current_path = modification.new_path or modification.old_path
                if not current_path:
                    logging.warning(f"Skipping modification with no path in commit {commit.hash[:8]}")
                    continue

                normalized_path = os.path.normpath(current_path)
                basename = os.path.basename(normalized_path)
                directory = os.path.dirname(normalized_path)
                
                logging.debug(f"Processing file: {normalized_path}")
                logging.debug(f"Change type: {modification.change_type}")

                if modification.change_type == ModificationType.ADD:
                    if normalized_path not in file_histories:
                        logging.debug(f"New file added: {normalized_path}")
                        file_histories[normalized_path] = FileHistory(
                            creation_date=commit.author_date,
                            full_path=normalized_path,
                            basename=basename,
                            directory=directory,
                            last_modified_date=commit.author_date,
                            is_deleted=False,
                        )
                        basename_map[basename].append(normalized_path)
                        logging.debug(f"Added to basename_map: {basename} -> {normalized_path}")

                elif modification.change_type == ModificationType.DELETE:
                    if normalized_path in file_histories:
                        logging.debug(f"Marking file as deleted: {normalized_path}")
                        file_histories[normalized_path].is_deleted = True

                elif modification.change_type == ModificationType.MODIFY:
                    if normalized_path in file_histories:
                        logging.debug(f"Updating last modified date for: {normalized_path}")
                        file_histories[normalized_path].last_modified_date = commit.author_date

        logging.info(f"Processed {commit_count} commits")
        logging.info(f"Found {len(file_histories)} unique files")
        logging.info(f"Created {len(basename_map)} basename mappings")

    except Exception as e:
        logging.error(f"Error processing repository: {str(e)}")
        raise

    return file_histories, basename_map

def match_tests_sources(file_histories: Dict[str, FileHistory], 
                       basename_map: Dict[str, List[str]]) -> List[MatchedPair]:
    logging.info("Starting test-source file matching process")
    matches = []
    test_pattern = re.compile(r'(.+)Test\.java$')
    
    logging.debug(f"Total files to process: {len(file_histories)}")
    
    test_files = {
        path: history 
        for path, history in file_histories.items() 
        if test_pattern.match(history.basename) and not history.is_deleted
    }
    
    logging.info(f"Found {len(test_files)} test files")
    
    for test_path, test_history in test_files.items():
        logging.debug(f"\nProcessing test file: {test_path}")
        match = test_pattern.match(test_history.basename)
        if not match:
            logging.warning(f"Test pattern match failed for: {test_history.basename}")
            continue
            
        source_basename = f"{match.group(1)}.java"
        logging.debug(f"Looking for source file: {source_basename}")
        
        potential_source_paths = basename_map.get(source_basename, [])
        logging.debug(f"Found {len(potential_source_paths)} potential source files")
        
        if not potential_source_paths:
            logging.warning(f"No source file found for test: {test_path}")
            continue
            
        if len(potential_source_paths) == 1:
            source_path = potential_source_paths[0]
            logging.debug(f"Single source file match found: {source_path}")
            
            if source_path in file_histories and not file_histories[source_path].is_deleted:
                matches.append(MatchedPair(
                    test_file=test_history,
                    source_file=file_histories[source_path],
                    directory_match=test_history.directory == file_histories[source_path].directory
                ))
                logging.debug("Match added to results")
        else:
            logging.debug("Multiple potential source files found, looking for best match")
            best_match = None
            for source_path in potential_source_paths:
                if source_path not in file_histories or file_histories[source_path].is_deleted:
                    logging.debug(f"Skipping deleted/missing source file: {source_path}")
                    continue
                    
                source_history = file_histories[source_path]
                
                if test_history.directory == source_history.directory:
                    logging.debug(f"Found exact directory match: {source_path}")
                    best_match = source_history
                    break
                    
                if is_related_directory(test_history.directory, source_history.directory):
                    logging.debug(f"Found related directory match: {source_path}")
                    best_match = source_history
            
            if best_match:
                matches.append(MatchedPair(
                    test_file=test_history,
                    source_file=best_match,
                    directory_match=test_history.directory == best_match.directory
                ))
                logging.debug("Best match added to results")
            else:
                logging.warning(f"No suitable match found for test file: {test_path}")
    
    logging.info(f"Total matches found: {len(matches)}")
    return matches

def is_related_directory(test_dir: str, source_dir: str) -> bool:
    logging.debug(f"Checking directory relationship between:\nTest: {test_dir}\nSource: {source_dir}")
    
    test_components = test_dir.split(os.sep)
    source_components = source_dir.split(os.sep)
    
    patterns = [("test", "main"), ("tests", "src"), ("test", "src"), ("tests", "main")]
    
    for test_pattern, source_pattern in patterns:
        test_normalized = [c if c != test_pattern else source_pattern for c in test_components]
        if test_normalized == source_components:
            logging.debug(f"Found related directories with pattern: {test_pattern} -> {source_pattern}")
            return True
    
    logging.debug("Directories are not related")
    return False

def analyze_tdd_patterns(matches: List[MatchedPair]) -> dict:
    logging.info("Starting TDD pattern analysis")
    logging.debug(f"Analyzing {len(matches)} matched pairs")
    
    results = {
        "total_matches": len(matches),
        "same_directory_matches": sum(1 for m in matches if m.directory_match),
        "test_first": 0,
        "test_after": 0,
        "same_commit": 0,
    }

    for i, match in enumerate(matches, 1):
        test_date = match.test_file.creation_date
        source_date = match.source_file.creation_date
        
        logging.debug(f"\nAnalyzing match {i}/{len(matches)}")
        logging.debug(f"Test file: {match.test_file.full_path}")
        logging.debug(f"Source file: {match.source_file.full_path}")
        logging.debug(f"Test creation date: {test_date}")
        logging.debug(f"Source creation date: {source_date}")

        if test_date < source_date:
            results["test_first"] += 1
            logging.debug("Pattern: Test First")
        elif test_date > source_date:
            results["test_after"] += 1
            logging.debug("Pattern: Test After")
        else:
            results["same_commit"] += 1
            logging.debug("Pattern: Same Commit")

    logging.info("Analysis Results:")
    for key, value in results.items():
        logging.info(f"{key}: {value}")
    
    return results

def analyze_repos(clone_dir):
    logging.info(f"Starting repository analysis in directory: {clone_dir}")
    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(OUTPUT_DIR, f"claude_tdd_results_{timestamp}.csv")

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        logging.debug(f"Created output directory: {OUTPUT_DIR}")

    for repo_name in os.listdir(clone_dir):
        repo_path = os.path.join(clone_dir, repo_name)
        if os.path.isdir(repo_path):
            logging.info(f"\nAnalyzing repository: {repo_name}")

            try:
                logging.info("Getting file histories...")
                file_histories, basename_map = get_file_creation_dates(repo_path)
                logging.debug(f"Found {len(file_histories)} files and {len(basename_map)} basenames")

                logging.info("Matching test and source files...")
                matches = match_tests_sources(file_histories, basename_map)
                logging.debug(f"Found {len(matches)} matched pairs")

                logging.info("Analyzing TDD patterns...")
                tdd_analysis = analyze_tdd_patterns(matches)

                repo_result = {
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
                
                logging.info("Repository Results:")
                for key, value in repo_result.items():
                    logging.info(f"{key}: {value}")
                
                results.append(repo_result)

            except Exception as e:
                logging.error(f"Error analyzing repository {repo_name}: {str(e)}")

    try:
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False)
        logging.info(f"Analysis complete. Results saved to {output_file}")
    except Exception as e:
        logging.error(f"Error saving results to CSV: {str(e)}")

def main():
    logging.info("Starting TDD analysis script")
    try:
        clone_repos(REPO_URLS, CLONE_DIR)
        analyze_repos(CLONE_DIR)
    except Exception as e:
        logging.error(f"Fatal error in main execution: {str(e)}")
    logging.info("Script execution completed")

if __name__ == "__main__":
    main()