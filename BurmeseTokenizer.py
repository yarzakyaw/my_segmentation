from typing import List, Tuple
from Trie import Trie
from utils import syllable_split
from math import log
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
import heapq
import json

class BurmeseTokenizer:
    def __init__(self, context_aware_dict_path: str, bpe_vocab_size: int = 10000):
        """
        Initialize the Burmese tokenizer with a context-aware dictionary.
        
        Args:
            context_aware_dict_path: Path to context_aware_dictionary.json.
            bpe_vocab_size: Vocabulary size for BPE training.
        """
        with open(context_aware_dict_path, 'r', encoding='utf-8') as f:
            self.dictionary = json.load(f)
        
        self.bpe_tokenizer = None
        self.bpe_vocab_size = bpe_vocab_size
        self.trie = Trie()
        for word, data in self.dictionary.items():
            self.trie.insert(
                word=word,
                freq=data["freq"],
                preceding=data["preceding"],
                following=data["following"]
            )

    def segment_syllables(self, text: str) -> List[str]:
        """Segment Burmese text into syllables using syllable_split."""
        return syllable_split(text)
    # Reusing context_aware_maximum_matching from previous implementation
    def context_aware_maximum_matching(self, syllables: List[str], beam_size: int = 3, use_context: bool = False) -> List[str]:
        """
        Recombine syllables into root words and particles using probability-based maximum matching.
        
        Args:
            syllables (List[str]): List of syllables.
            beam_size (int): Number of segmentations to consider (for disambiguation).
            use_context: False
        
        Returns:
            List[str]: List of tokenized root words and particles.
        """
        
        beam = [(0, [], 0.0, None)]  # (position, tokens, score, last_token)
        final_segmentations = []

        while beam:
            new_beam = []
            for pos, tokens, score, last_token in beam:
                if pos >= len(syllables):
                    final_segmentations.append((tokens, score))
                    continue
                matches = self.trie.find_all_matches(syllables, pos)
                if not matches:
                    matches = [(syllables[pos], 0.5, pos + 1, [], [])]
                
                for token, freq, next_pos, preceding, following in matches:
                    new_score = score + log(max(freq, 1e-10)) - 0.01 * len(token)
                    
                    if use_context and last_token and tokens:
                        for prev in preceding:
                            if prev["token"] == last_token:
                                new_score += log(max(prev["prob"], 1e-10))
                                break
                        prev_matches = self.trie.find_all_matches(syllables, pos - 1)
                        for _, _, _, _, prev_following in prev_matches:
                            for next_t in prev_following:
                                if next_t["token"] == token:
                                    new_score += log(max(next_t["prob"], 1e-10))
                                    break
                    
                    new_tokens = tokens + [token]
                    new_beam.append((next_pos, new_tokens, new_score, token))
            
            beam = heapq.nlargest(beam_size, new_beam, key=lambda x: x[2])

        if final_segmentations:
            best_tokens, _ = max(final_segmentations, key=lambda x: x[1])
            return best_tokens
        return syllables
    
    def train_bpe(self, texts: List[str], special_tokens: List[str] = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]):
        """
        Train a BPE tokenizer with dictionary constraints on a raw Burmese text corpus.
        
        Args:
            texts (List[str]): List of raw Burmese texts (e.g., sentences from your dataset).
            special_tokens (List[str]): Special tokens for NLP frameworks.
        """
        self.bpe_tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
        self.bpe_tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        self.bpe_tokenizer.decoder = decoders.ByteLevel()

        # Pre-tokenize corpus using context aware maximum_matching to protect dictionary tokens
        pretokenized_texts = []
        for text in texts:
            syllables = self.segment_syllables(text)
            tokens = self.context_aware_maximum_matching(syllables, use_context = True)
            pretokenized_texts.append(' '.join(tokens))

        # Add dictionary tokens as protected tokens
        protected_tokens = list(self.dictionary.keys()) + special_tokens
        trainer = trainers.BpeTrainer(
            vocab_size=self.bpe_vocab_size,
            special_tokens=protected_tokens,
            initial_alphabet=[c for word in self.dictionary.keys() for c in word]
        )
        self.bpe_tokenizer.train_from_iterator(pretokenized_texts, trainer)
        self.bpe_tokenizer.save("burmese_bpe_tokenizer.json")
    
    def tokenize(self, text: str, use_bpe: bool = True) -> Tuple[List[str], List[int]]:
        """
        Tokenize Burmese text into sub-word units.
        
        Args:
            text (str): Input Burmese text.
            use_bpe (bool): Whether to apply BPE after dictionary-based tokenization.
        
        Returns:
            Tuple[List[str], List[int]]: List of tokens and their corresponding IDs.
        """
        syllables = self.segment_syllables(text)
        tokens = self.context_aware_maximum_matching(syllables, use_context = True)
        if use_bpe and self.bpe_tokenizer:
            encoded = self.bpe_tokenizer.encode(' '.join(tokens))
            return encoded.tokens, encoded.ids
        return tokens, list(range(len(tokens)))
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to Burmese text."""
        if self.bpe_tokenizer:
            return self.bpe_tokenizer.decode(token_ids)
        return ''.join(self.dictionary.get(id, '[UNK]') for id in token_ids)

    def batch_tokenize(self, texts: List[str], use_bpe: bool = True) -> List[Tuple[List[str], List[int]]]:
        """Tokenize a batch of texts for efficiency."""
        return [self.tokenize(text, use_bpe) for text in texts]