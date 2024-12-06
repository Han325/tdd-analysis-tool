import re
from dataclasses import dataclass
from datetime import datetime
from typing import List, Tuple
from FileContent import FileContent

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