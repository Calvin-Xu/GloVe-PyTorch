import json
from typing import Dict
import numpy as np


def word2vec(word: str, embeddings: np.ndarray, vocab: Dict[str, int]) -> np.ndarray:
    """
    Returns the vector representation of the given word.

    Args:
        word: Word to be represented.
        embeddings: Embeddings matrix.
        vocab: Vocabulary.

    Returns:
        Embedding of the given word.
    """
    return embeddings[vocab[word]]


def most_similar_words(
    input: np.ndarray, embeddings: np.ndarray, idx_to_vocab: Dict[int, str], k: int = 10
) -> np.ndarray:
    """
    Returns the k most similar vectors to the input vector.

    Args:
        input: Input vector.
        embeddings: Embeddings matrix.
        vocab: Vocabulary.
        k: Number of similar vectors to return.

    Returns:
        k most similar vectors to the input vector.
    """
    similarities = np.dot(embeddings, input) / np.linalg.norm(embeddings, axis=1)
    indices = np.argsort(similarities)[::-1][:k]
    return embeddings[indices], [idx_to_vocab[str(idx)] for idx in indices]


if __name__ == "__main__":
    embeddings = np.load("final_embeddings.npy")
    vocab = json.load(open("vocab.json"))
    idx_to_vocab = json.load(open("idx_to_vocab.json"))

    king = word2vec("king", embeddings, vocab)
    man = word2vec("man", embeddings, vocab)
    woman = word2vec("woman", embeddings, vocab)
    similar_vectors, similar_words = most_similar_words(
        king - man + woman, embeddings, idx_to_vocab
    )

    print("Similar words:", similar_words)
