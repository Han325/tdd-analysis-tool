import os
import re
import pandas as pd
from pydriller import Repository
from git import Repo

# List of repository URLs
REPO_URLS = [
    "https://github.com/apache/bigtop-manager",
    # Add more URLs here
]

# Directory to clone repositories
CLONE_DIR = "apache_repos"

# Bot authors to exclude
BOT_AUTHORS = ["dependabot[bot]", "github-actions[bot]", "renovate[bot]", "snyk-bot"]

# Test file patterns
TEST_FILE_PATTERNS = [r'.*Test\.java$', r'.*test_.*\.py$']
SOURCE_FILE_PATTERNS = [r'.*\.java$', r'.*\.py$']

def is_bot_commit(commit):
    return any(bot.lower() in commit.author.name.lower() for bot in BOT_AUTHORS)

def is_test_file(filename):
    return any(re.match(pattern, filename) for pattern in TEST_FILE_PATTERNS)

def is_source_file(filename):
    return any(re.match(pattern, filename) for pattern in SOURCE_FILE_PATTERNS)

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

def analyze_repos(clone_dir):
    results = []
    for repo_name in os.listdir(clone_dir):
        repo_path = os.path.join(clone_dir, repo_name)
        if os.path.isdir(repo_path):
            test_commits = 0
            code_commits = 0
            for commit in Repository(repo_path).traverse_commits():
                if is_bot_commit(commit):
                    continue
                added_tests = any(is_test_file(mod.filename) and mod.added_lines > 0 for mod in commit.modified_files)
                added_code = any(not is_test_file(mod.filename) and is_source_file(mod.filename) and mod.added_lines > 0 for mod in commit.modified_files)
                if added_tests:
                    test_commits += 1
                if added_code:
                    code_commits += 1
            tdd_ratio = test_commits / (code_commits + 1)
            results.append({
                'repository': repo_name,
                'test_commits': test_commits,
                'code_commits': code_commits,
                'tdd_ratio': tdd_ratio
            })
    df = pd.DataFrame(results)
    df.to_csv("tdd_analysis_results.csv", index=False)
    print("Analysis complete. Results saved to tdd_analysis_results.csv")

def main():
    clone_repos(REPO_URLS, CLONE_DIR)
    analyze_repos(CLONE_DIR)

if __name__ == "__main__":
    main()