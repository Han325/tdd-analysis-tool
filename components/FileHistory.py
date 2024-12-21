from dataclasses import dataclass
from typing import List, Set, Tuple, Optional
from components.FileContent import FileContent
from datetime import datetime

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