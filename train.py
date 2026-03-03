"""
Training script — Neural Network from Scratch on MNIST
=======================================================

Run:
    python train.py

Hyperparameters (edit at the top of main()):
    epochs        — full passes over the training set
    learning_rate — step size for gradient descent
    batch_size    — samples per gradient update
    hidden_size   — neurons in the hidden layer
"""

import numpy as np
from neural_network import (
    init_params,
    init_adam_state,
    forward_pass,
    backward_pass,
    update_params,
    adam_update,
    accuracy,
    cross_entropy_loss,
)
from utils import load_mnist, one_hot_encode, get_mini_batches


def train(X_train, Y_train, X_test, Y_test,
          epochs=20,
          learning_rate=None,
          batch_size=64,
          hidden_size=128,
          optimizer="adam"):
    """
    Mini-batch training loop supporting SGD and Adam.

    Parameters
    ----------
    optimizer     : "adam" (default) or "sgd"
    learning_rate : defaults to 0.001 for Adam, 0.1 for SGD if not specified

    Prints loss and accuracy after every epoch.
    Returns (params, history) where history is a dict of lists.
    """
    if learning_rate is None:
        learning_rate = 0.001 if optimizer == "adam" else 0.1

    Y_train_oh = one_hot_encode(Y_train)
    params = init_params(hidden_size=hidden_size)
    adam_state = init_adam_state(params) if optimizer == "adam" else None
    rng = np.random.default_rng(0)

    history = {"train_loss": [], "train_acc": [], "test_acc": []}

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        num_batches = 0

        for X_batch, Y_batch in get_mini_batches(X_train, Y_train_oh, batch_size, rng):
            A2, cache = forward_pass(X_batch, params)
            epoch_loss += cross_entropy_loss(A2, Y_batch)
            grads = backward_pass(X_batch, Y_batch, params, cache)

            if optimizer == "adam":
                params, adam_state = adam_update(params, grads, adam_state, lr=learning_rate)
            else:
                params = update_params(params, grads, learning_rate)

            num_batches += 1

        avg_loss  = epoch_loss / num_batches
        train_acc = accuracy(X_train, Y_train, params)
        test_acc  = accuracy(X_test,  Y_test,  params)

        history["train_loss"].append(avg_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)

        print(
            f"Epoch {epoch:>3}/{epochs}  "
            f"loss: {avg_loss:.4f}  "
            f"train acc: {train_acc:.4f}  "
            f"test acc: {test_acc:.4f}"
        )

    return params, history


def main():
    # ---- Hyperparameters ----
    epochs        = 20
    learning_rate = None   # None → uses optimizer default (0.001 Adam, 0.1 SGD)
    batch_size    = 64
    hidden_size   = 128
    optimizer     = "adam"
    # -------------------------

    X_train, X_test, Y_train, Y_test = load_mnist()

    print(f"\nTraining set : {X_train.shape[1]:,} samples")
    print(f"Test set     : {X_test.shape[1]:,} samples")
    print(f"Input size   : {X_train.shape[0]} features (28×28 pixels)")
    print(f"\nArchitecture : 784 → {hidden_size} (ReLU) → 10 (Softmax)")
    print(f"Optimizer    : {optimizer.upper()}")
    print()

    params, history = train(
        X_train, Y_train, X_test, Y_test,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        hidden_size=hidden_size,
        optimizer=optimizer,
    )

    final_acc = accuracy(X_test, Y_test, params)
    print(f"\nFinal test accuracy: {final_acc * 100:.2f}%")

    np.savez("trained_params.npz", **params)
    print("Parameters saved to trained_params.npz")


if __name__ == "__main__":
    main()
