from dataclasses import dataclass
from typing import List

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