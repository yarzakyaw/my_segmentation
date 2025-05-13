from typing import List, Dict, Tuple

# Trie implementation for efficient dictionary lookups
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.token = None
        self.freq = 0.0
        self.preceding = []  # List of {"token": str, "prob": float}
        self.following = []  # List of {"token": str, "prob": float}
        
class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str, freq: float = 1.0, preceding: List[Dict] = None, following: List[Dict] = None):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True
        node.token = word
        node.freq = freq
        node.preceding = preceding or []
        node.following = following or []

    def find_all_matches(self, syllables: List[str], start: int) -> List[Tuple[str, float, int, List[Dict], List[Dict]]]:
        matches = []
        max_len = min(10, len(syllables) - start)
        for length in range(1, max_len + 1):
            candidate = ''.join(syllables[start:start + length])
            node = self.root
            valid = True
            for char in candidate:
                if char not in node.children:
                    valid = False
                    break
                node = node.children[char]
            if valid and node.is_end:
                matches.append((node.token, node.freq, start + length, node.preceding, node.following))
        return matches