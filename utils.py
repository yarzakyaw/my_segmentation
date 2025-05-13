from typing import List, Dict, Tuple
from collections import defaultdict
from tqdm import tqdm
import json
import regex as re

def get_syllable(label: str, burmese_consonant: str, others: str) -> List[str]:
    """
    Segment a Burmese word into syllables using regex-based rules.
    
    Args:
        label (str): Input Burmese text or word.
        burmese_consonant (str): Regex range for Burmese consonants.
        others (str): Regex range for other characters (vowels, punctuation, etc.).
    
    Returns:
        List[str]: List of syllables.
    """
    # Define regex patterns for Burmese consonants and other characters
    # label = re.sub(r"(?<![္])(["+burmese_consonant+"])(?![်္|့])|(["+others+"])", r" \1\2", label).strip()
    # label = re.sub('(?<=[က-ၴ])([a-zA-Z0-9])', r' \1', label)
    # label = re.sub('([0-9၀-၉])\s+([0-9၀-၉])\s*', r'\1\2 ', label)
    # label = re.sub('([0-9၀-၉])\s+(\+)', r'\1 \2 ', label)
    # label = label.split()
    label = re.sub(r"(?<![္])([" + burmese_consonant + r"])(?![်္|့])|([" + others + r"])", r" \1\2", label).strip()
    label = re.sub(r"(?<=[က-ၴ])([a-zA-Z0-9])", r" \1", label)
    label = re.sub(r"([0-9၀-၉])\s+([0-9၀-၉])\s*", r"\1\2 ", label)
    label = re.sub(r"([0-9၀-၉])\s+(\+)", r"\1 \2 ", label)
    label = label.split()
    
    return label

def syllable_split(label: str) -> List[str]:
    """
    Split Burmese text into syllables, handling spaces and word boundaries.
    
    Args:
        label (str): Input Burmese text.
    
    Returns:
        List[str]: List of syllables.
    """
    burmese_consonant = 'က-အ'
    others = r"ဣဤဥဦဧဩဪဿ၌၍၏၀-၉၊။!-/:-@[-`{-~\s.,"
    
    label_syllable = [get_syllable(s, burmese_consonant, others) + [' '] for s in label.split()]
    return [s for sublist in label_syllable for s in sublist][:-1]

def build_context_aware_frequency_dictionary(
    tokenized_corpus: List[List[str]],
    initial_dictionary: Dict[str, str],
    output_path: str = "context_aware_dictionary.json",
    top_k: int = 3,
    min_freq: float = 1e-5
) -> Dict[str, Dict]:
    """
    Build a context-aware frequency dictionary from a pre-tokenized Burmese corpus.
    
    Args:
        tokenized_corpus: List of tokenized sentences (e.g., [["ပညာရေး", "ဝန်ကြီးဌာန"], ...]).
        initial_dictionary: Initial dictionary mapping tokens to parts (e.g., {"ပညာရေး": "root"}).
        output_path: Path to save the context-aware dictionary as JSON.
        top_k: Number of preceding/following tokens to store per token.
        min_freq: Minimum frequency for smoothing unseen tokens.
    
    Returns:
        Dict[str, Dict]: Context-aware dictionary with part, freq, preceding, and following.
    """
    # Initialize counters
    token_counts = defaultdict(int)
    preceding_counts = defaultdict(lambda: defaultdict(int))  # (token, prev) -> count
    following_counts = defaultdict(lambda: defaultdict(int))  # (token, next) -> count
    total_tokens = 0

    # Special tokens for sentence boundaries
    START_TOKEN = "<START>"
    END_TOKEN = "<END>"

    # Process pre-tokenized corpus
    for tokens in tqdm(tokenized_corpus, desc="Processing tokenized corpus"):
        # Validate tokens
        for token in tokens:
            if token not in initial_dictionary:
                print(f"Warning: Token '{token}' not in dictionary, skipping.")
                continue
            token_counts[token] += 1
            total_tokens += 1
        
        # Count transitions
        for i, token in enumerate(tokens):
            if token not in initial_dictionary:
                continue
            # Preceding token
            prev_token = START_TOKEN if i == 0 else tokens[i-1]
            if prev_token in initial_dictionary or prev_token == START_TOKEN:
                preceding_counts[token][prev_token] += 1
            # Following token
            next_token = END_TOKEN if i == len(tokens) - 1 else tokens[i+1]
            if next_token in initial_dictionary or next_token == END_TOKEN:
                following_counts[token][next_token] += 1

    # Compute probabilities with smoothing
    vocab_size = len(initial_dictionary) + 2  # Include <START> and <END>
    result = {}

    for token in initial_dictionary:
        # Token frequency with smoothing
        count = token_counts.get(token, 0)
        freq = (count + 1) / (total_tokens + vocab_size)

        # Preceding tokens
        preceding_probs = []
        total_preceding = sum(preceding_counts[token].values())
        for prev_token, prev_count in preceding_counts[token].items():
            prob = (prev_count + 1) / (total_preceding + vocab_size)
            preceding_probs.append({"token": prev_token, "prob": prob})
        preceding_probs = sorted(preceding_probs, key=lambda x: x["prob"], reverse=True)[:top_k]

        # Following tokens
        following_probs = []
        total_following = sum(following_counts[token].values())
        for next_token, next_count in following_counts[token].items():
            prob = (next_count + 1) / (total_following + vocab_size)
            following_probs.append({"token": next_token, "prob": prob})
        following_probs = sorted(following_probs, key=lambda x: x["prob"], reverse=True)[:top_k]

        result[token] = {
            "part": initial_dictionary[token],
            "freq": max(freq, min_freq),
            "preceding": preceding_probs,
            "following": following_probs
        }

    # Save dictionary
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result