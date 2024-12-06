from FileHistory import FileHistory
from typing import NamedTuple

class MatchedPair(NamedTuple):
    test_file: FileHistory
    source_file: FileHistory
    directory_match: bool
    confidence_score: float  # Indicates confidence in the match
    relationship_type: str  # 'direct', 'abstract', 'utility', etc.