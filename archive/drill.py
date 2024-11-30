import os
import random
import re
import pandas as pd
from datetime import datetime
from pydriller import Repository, ModificationType
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
TEST_FILE_PATTERNS = [r".*Test\.java$", r".*test_.*\.py$"]
SOURCE_FILE_PATTERNS = [r".*\.java$", r".*\.py$"]


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


def get_file_creation_dates(repo_path):
    file_creation_dict = {}
    rename_mapping = {}

    branch_to_analyze = "main"

    for commit in Repository(
        repo_path, only_in_branch=branch_to_analyze
    ).traverse_commits():
        for modification in commit.modified_files:
            full_path = modification.new_path or modification.old_path
            if not full_path:
                continue

            filename = os.path.basename(full_path)

            if modification.change_type == ModificationType.ADD:
                if filename not in file_creation_dict:
                    file_creation_dict[filename] = commit.author_date

            elif modification.change_type == ModificationType.RENAME:
                old_full_path = modification.old_path
                if old_full_path:
                    original_filename = os.path.basename(old_full_path)
                    original_creation_date = file_creation_dict.get(original_filename)
                    if original_creation_date:
                        file_creation_dict[filename] = original_creation_date
                        rename_mapping[filename] = original_filename

    file_creation_tuples = [(date, fname) for fname, date in file_creation_dict.items()]

    sample_size = 10
    actual_sample_size = min(sample_size, len(file_creation_tuples))
    random_samples = (
        random.sample(file_creation_tuples, actual_sample_size)
        if file_creation_tuples
        else []
    )

    if random_samples:
        print("\n--- Sample File Creation Dates (Random 10 Files) ---")
        print("{:<5} {:<50} {:<30}".format("No.", "Filename", "Creation Date"))
        print("-" * 90)
        for idx, (date, fname) in enumerate(random_samples, start=1):
            formatted_date = (
                date.strftime("%Y-%m-%d %H:%M:%S %Z")
                if isinstance(date, datetime)
                else date
            )
            print("{:<5} {:<50} {:<30}".format(idx, fname, formatted_date))
        print("-----------------------------------------------------\n")
    else:
        print("\nNo files found in the repository to display.\n")
    return file_creation_tuples


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

    # # Print the source files
    # print("\nSource Files:")
    # for source_file, date in sources.items():
    #     print(f"{source_file} - Created on {date}")

    return sources, tests


def match_tests_sources(sources, tests):
    matches = []
    for test_file, test_date in tests.items():
        # Example matching: ClassTest.java -> Class.java
        match = re.match(r"(.+)Test\.java$", test_file)
        if match:
            source_file = match.group(1) + ".java"
            if source_file in sources:
                matches.append(
                    {
                        "test_file": test_file,
                        "source_file": source_file,
                        "test_creation": test_date,
                        "source_creation": sources[source_file],
                    }
                )

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

            results.append(
                {
                    "repository": repo_name,
                    "total_test_files": total_tests,
                    "total_source_files": total_sources,
                    "matched_test_source": matched,
                    "tdd_ratio": tdd_ratio,
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
