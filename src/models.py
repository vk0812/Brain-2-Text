"""fMRI → image-embedding models (beyond the sklearn ridge baseline)."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """One hidden layer: voxels → hidden → embedding.

    With ~4k training trials and ~1.5k voxels the model can easily memorise the
    training set, so dropout (on the input voxels too) and weight decay do most of
    the work of keeping it general.
    """

    def __init__(self, n_voxels, embedding_dim, hidden_dim=1024, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(n_voxels, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, x):
        return self.net(x)


def cosine_loss(predicted, target):
    """1 - cosine similarity: optimises the same quantity retrieval uses to rank."""
    return (1 - F.cosine_similarity(predicted, target, dim=1)).mean()


def infonce_loss(predicted, target, labels, temperature):
    """Symmetric contrastive (CLIP-style) loss over a batch.

    Row i of the similarity matrix asks "which image in the batch did trial i
    see?"; the answer is column i, every other image is a negative. Column i asks
    the reverse question. Trials of the *same* image elsewhere in the batch are not
    real negatives, so their logits are masked out.
    """
    predicted = F.normalize(predicted, dim=1)
    target = F.normalize(target, dim=1)
    logits = predicted @ target.T / temperature

    same_image = labels[:, None] == labels[None, :]
    off_diagonal = ~torch.eye(len(labels), dtype=torch.bool)
    logits = logits.masked_fill(same_image & off_diagonal, float("-inf"))

    answer = torch.arange(len(labels))
    return (F.cross_entropy(logits, answer) + F.cross_entropy(logits.T, answer)) / 2


def train_mlp(X, Y, epochs, weight_decay, seed, lr=1e-3, batch_size=128, on_epoch_end=None,
              loss="cosine", labels=None, temperature=0.1):
    """Train an MLP for a fixed number of epochs.

    loss: "cosine" (regress towards the target) or "infonce" (contrastive; needs
    `labels`, one integer image id per row, to mask same-image pairs).
    on_epoch_end(epoch, model) is called after each epoch, e.g. to track a
    validation score and pick the best epoch count.
    """
    torch.manual_seed(seed)
    X = torch.as_tensor(X, dtype=torch.float32)
    Y = torch.as_tensor(Y, dtype=torch.float32)
    if loss == "infonce":
        labels = torch.as_tensor(labels)
    model = MLP(X.shape[1], Y.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    generator = torch.Generator().manual_seed(seed)

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in torch.randperm(len(X), generator=generator).split(batch_size):
            optimizer.zero_grad()
            if loss == "infonce":
                loss_value = infonce_loss(model(X[batch]), Y[batch], labels[batch], temperature)
            else:
                loss_value = cosine_loss(model(X[batch]), Y[batch])
            loss_value.backward()
            optimizer.step()
        if on_epoch_end is not None:
            on_epoch_end(epoch, model)
    return model


def predict_mlp(model, X):
    model.eval()
    with torch.inference_mode():
        return model(torch.as_tensor(X, dtype=torch.float32)).numpy()
