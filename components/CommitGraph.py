import networkx as nx
from typing import List, Optional

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