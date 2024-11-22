import os
import re
import pandas as pd
from datetime import datetime
from pydriller import Repository
from git import Repo


# List of fucking repository URLs
REPO_URLS = [
    "https://github.com/apache/bigtop-manager",
]

# Directory to clone repositories
CLONE_DIR = "apache_repos"

# Directory to save analysis results
OUTPUT_DIR = "results"

# Patterns to identify test files (adjust as needed)
TEST_FILE_PATTERNS = [r'.*Test\.java$', r'.*test_.*\.py$']
SOURCE_FILE_PATTERNS = [r'.*\.java$', r'.*\.py$']


def clone_repos(repo_urls, clone_dir):
    if not os.path.exists(clone_dir):
        os.makedirs(clone_dir)
    for url in repo_urls:
        repo_name = url.split('/')[-1].replace('.git', '')
        repo_path = os.path.join(clone_dir, repo_name)
        if not os.path.exists(repo_path):
            print(f"Cloning {repo_name}...")
            Repo.clone_from(url, repo_path)
        else:
            print(f"{repo_name} already cloned.")

# TODO: fix the querying of the fucking file creation data
def get_file_creation_dates(repo_path):
    file_creation_list = list()
    for commit in Repository(repo_path).traverse_commits():
        for modification in commit.modified_files:
            filename = modification.filename
            if not any(filename == existing_filename for _, existing_filename in file_creation_list):
                if modification.source_code_before is None:
                    file_creation_list.append((commit.committer_date, filename))  # **CHANGED** to append tuple

    # print("FUCKING COUNT", len(file_creation_list))
    return file_creation_list

def classify_files(file_creation):
    tests = {}
    sources = {}
    for creation_date, filename in file_creation:
        if any(re.match(pattern, filename) for pattern in TEST_FILE_PATTERNS):
            tests[filename] = creation_date
        elif any(re.match(pattern, filename) for pattern in SOURCE_FILE_PATTERNS):
            sources[filename] = creation_date

    # # Print the test files
    # print("\nTest Files:")
    # for test_file, date in tests.items():
    #     print(f"{test_file} - Created on {date}")

    # Print the source files
    # print("\nSource Files:")
    # for source_file, date in sources.items():
    #     print(f"{source_file} - Created on {date}")

    return sources, tests

def match_tests_sources(sources, tests):
    matches = []
    for test_file, test_date in tests.items():
        # Example matching: ClassTest.java -> Class.java
        match = re.match(r'(.+)Test\.java$', test_file)
        if match:
            source_file = match.group(1) + ".java"
            if source_file in sources:
                matches.append({
                    'test_file': test_file,
                    'source_file': source_file,
                    'test_creation': test_date,
                    'source_creation': sources[source_file]
                })

    return matches

def analyze_repos(clone_dir):
    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(OUTPUT_DIR, f"tdd_results_{timestamp}.csv") 
    
    for repo_name in os.listdir(clone_dir):
        repo_path = os.path.join(clone_dir, repo_name)
        if os.path.isdir(repo_path):
            print(f"Analyzing {repo_name}...")
            file_creation = get_file_creation_dates(repo_path)
            sources, tests = classify_files(file_creation)
            matches = match_tests_sources(sources, tests)
            
            total_tests = len(tests)
            total_sources = len(sources)
            matched = len(matches)
            tdd_ratio = matched / (total_sources + 1)
            
            results.append({
                'repository': repo_name,
                'total_test_files': total_tests,
                'total_source_files': total_sources,
                'matched_test_source': matched,
                'tdd_ratio': tdd_ratio
            })
    
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False) 
    print(f"Analysis complete. Results saved to {output_file}")

def main():
    clone_repos(REPO_URLS, CLONE_DIR)
    analyze_repos(CLONE_DIR)

if __name__ == "__main__":
    main()