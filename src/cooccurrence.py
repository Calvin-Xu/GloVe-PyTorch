import re
import numpy as np
from collections import defaultdict
from datasets import load_dataset
from typing import List, Dict, Tuple
import json


def tokenize_and_build_vocab(
    corpus: List[str], vocab_size: int = None
) -> Tuple[List[List[str]], Dict[str, int], Dict[int, str]]:
    """
    Tokenizes the corpus and builds a vocabulary of the most frequent words.

    Args:
        corpus: List of sentences in the corpus.
        vocab_size: The maximum size of the vocabulary.

    Returns:
        tokenized_corpus: List of tokenized sentences.
        vocab: Dictionary mapping words to indices.
        idx_to_vocab: Dictionary mapping indices to words.
    """
    tokenized_corpus = []
    word_freq = defaultdict(int)

    for sentence in corpus:
        tokens = [word.lower() for word in sentence.strip().split()]
        tokenized_corpus.append(tokens)
        for token in tokens:
            if re.match("^[A-Za-z'-]+$", token):  # only match English words
                word_freq[token] += 1

    sorted_vocab = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)

    if vocab_size:
        sorted_vocab = sorted_vocab[:vocab_size]

    vocab = {word: idx for idx, (word, _) in enumerate(sorted_vocab)}
    idx_to_vocab = {idx: word for word, idx in vocab.items()}

    return tokenized_corpus, vocab, idx_to_vocab


def generate_cooccurrence_matrix(
    tokenized_corpus: List[List[str]], vocab: Dict[str, int], window_size: int = 5
) -> np.ndarray:
    """
    Generates a co-occurrence matrix from the tokenized corpus.

    Args:
        tokenized_corpus: List of tokenized sentences.
        vocab: Dictionary mapping words to indices.
        window_size: Size of the context window.

    Returns:
        cooccurrence_matrix: Numpy array of shape (vocab_size, vocab_size).
    """
    vocab_size = len(vocab)
    cooccurrence_matrix = np.zeros((vocab_size, vocab_size), dtype=np.float32)

    for tokens in tokenized_corpus:
        sentence_len = len(tokens)
        for i, token in enumerate(tokens):
            if token not in vocab:
                continue
            token_idx = vocab[token]

            start = max(0, i - window_size)
            end = min(sentence_len, i + window_size + 1)  # range is [start, end)

            for j in range(start, end):
                if i == j:
                    continue
                context_token = tokens[j]
                if context_token in vocab:
                    context_idx = vocab[context_token]
                    # update co-occurrence count w/ weighting
                    cooccurrence_matrix[token_idx, context_idx] += 1.0 / abs(j - i)

    return cooccurrence_matrix


if __name__ == "__main__":
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
    corpus = dataset["train"]["text"]

    tokenized_corpus, vocab, idx_to_vocab = tokenize_and_build_vocab(
        corpus, vocab_size=10000
    )

    cooccurrence_matrix = generate_cooccurrence_matrix(tokenized_corpus, vocab)

    np.save("cooccurrence_matrix.npy", cooccurrence_matrix)

    with open("vocab.json", "w") as f:
        json.dump(vocab, f)

    with open("idx_to_vocab.json", "w") as f:
        json.dump(idx_to_vocab, f)

    print("Co-occurrence matrix and vocabulary have been saved.")
