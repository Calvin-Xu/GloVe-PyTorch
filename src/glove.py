import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple

# use mps on Mac
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class GloVe(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int) -> None:
        """
        Initialize the GloVe model.

        Args:
            vocab_size: Size of the vocabulary.
            embedding_dim: Dimensionality of the embeddings.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)

        self.word_biases = nn.Embedding(vocab_size, 1)
        self.context_biases = nn.Embedding(vocab_size, 1)

        nn.init.xavier_uniform_(self.word_embeddings.weight)
        nn.init.xavier_uniform_(self.context_embeddings.weight)
        nn.init.zeros_(self.word_biases.weight)
        nn.init.zeros_(self.context_biases.weight)

    def forward(
        self, word_indices: torch.Tensor, context_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the GloVe model's prediction for the given word-context pairs.

        Args:
            word_indices: Tensor of shape (batch_size,)
            context_indices: Tensor of shape (batch_size,)

        Returns:
            Tensor of shape (batch_size,) containing the predictions.
        """
        word_embedding = self.word_embeddings(
            word_indices
        )  # shape: (batch_size, embedding_dim)
        context_embedding = self.context_embeddings(
            context_indices
        )  # shape: (batch_size, embedding_dim)
        word_bias = self.word_biases(word_indices).squeeze()  # shape: (batch_size,)
        context_bias = self.context_biases(
            context_indices
        ).squeeze()  # shape: (batch_size,)

        # word_embedding * context_embedding shape: (batch_size, embedding_dim)
        dot_product = torch.sum(
            word_embedding * context_embedding, dim=1
        )  # shape: (batch_size,)

        return dot_product + word_bias + context_bias

    def loss(
        self,
        predictions: torch.Tensor,
        log_Xij: torch.Tensor,
        weighting: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the GloVe loss for the given predictions.

        Args:
            predictions: Tensor of shape (batch_size,)
            log_Xij: Tensor of shape (batch_size,)
            weighting: Tensor of shape (batch_size,)

        Returns:
            Scalar tensor representing the loss.
        """
        return torch.mean(weighting * (predictions - log_Xij) ** 2)


class CooccurrenceDataset(Dataset):
    def __init__(
        self, cooccurrence_matrix: np.ndarray, x_max: float, alpha: float
    ) -> None:
        """
        Dataset for non-zero entries in the co-occurrence matrix.

        Args:
            cooccurrence_matrix: numpy array of shape (vocab_size, vocab_size)
            x_max: maximum value of X_ij to consider in weighting function
            alpha: exponent parameter for the weighting function
        """
        nonzero_indices = np.nonzero(cooccurrence_matrix)
        self.word_indices = torch.from_numpy(nonzero_indices[0]).long()
        self.context_indices = torch.from_numpy(nonzero_indices[1]).long()
        Xij_values = cooccurrence_matrix[
            nonzero_indices
        ].astype(
            np.float32
        )  # shape: (num_nonzero_entries = len(nonzero_indices[0]) = len(nonzero_indices[1]),)

        # Compute weighting and log(Xij)
        self.Xij_values = torch.from_numpy(Xij_values)
        self.log_Xij_values = torch.log(self.Xij_values)

        # Compute weighting function f(Xij)
        weight = torch.pow(self.Xij_values / x_max, alpha)
        self.weighting = torch.minimum(weight, torch.ones_like(weight))

    def __len__(self) -> int:
        return len(self.Xij_values)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get the word index, context index, log(X_ij), and weighting at the given index.

        Returns:
            word_index: Tensor scalar
            context_index: Tensor scalar
            log_Xij: Tensor scalar
            weighting: Tensor scalar
        """
        return (
            self.word_indices[idx],
            self.context_indices[idx],
            self.log_Xij_values[idx],
            self.weighting[idx],
        )


def train(
    cooccurrence_matrix: np.ndarray,
    vocab_size: int,
    embedding_dim: int,
    batch_size: int = 512,
    epochs: int = 100,
    lr: float = 0.001,
    x_max: float = 100.0,
    alpha: float = 0.75,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Train the GloVe model.

    Args:
        cooccurrence_matrix: Numpy array of shape (vocab_size, vocab_size)
        vocab_size: Size of the vocabulary
        embedding_dim: Dimensionality of the embeddings
        batch_size: Batch size for training
        epochs: Number of training epochs
        lr: Learning rate
        x_max: Parameter for the weighting function
        alpha: Exponent parameter for the weighting function

    Returns:
        word_embeddings: Tensor of shape (vocab_size, embedding_dim)
        context_embeddings: Tensor of shape (vocab_size, embedding_dim)
    """
    model = GloVe(vocab_size, embedding_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    dataset = CooccurrenceDataset(cooccurrence_matrix, x_max, alpha)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            word_indices, context_indices, log_Xij_values, weighting_values = [
                x.to(device) for x in batch
            ]

            optimizer.zero_grad()
            predictions = model(word_indices, context_indices)
            loss = model.loss(predictions, log_Xij_values, weighting_values)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    return model.word_embeddings.weight.data, model.context_embeddings.weight.data


if __name__ == "__main__":
    cooccurrence_matrix = np.load("cooccurrence_matrix.npy")
    vocab_size = cooccurrence_matrix.shape[0]
    embedding_dim = 300

    word_embeddings, context_embeddings = train(
        cooccurrence_matrix,
        vocab_size,
        embedding_dim,
        batch_size=2**20,
        epochs=100,
        lr=0.001,
        x_max=100,
        alpha=0.75,
    )

    final_embeddings = word_embeddings + context_embeddings

    np.save("word_embeddings.npy", word_embeddings.cpu().numpy())
    np.save("context_embeddings.npy", context_embeddings.cpu().numpy())
    np.save("final_embeddings.npy", final_embeddings.cpu().numpy())
    print("Training complete. Word, context, and final embeddings saved to disk.")
