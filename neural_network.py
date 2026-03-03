"""
Neural Network from Scratch
============================
A fully-connected neural network implemented using only NumPy.

Architecture:
  Input layer  : 784 neurons  (28x28 flattened MNIST pixels)
  Hidden layer : 128 neurons  (ReLU activation)
  Output layer :  10 neurons  (Softmax activation, one per digit 0-9)

Forward pass:
  Z1 = W1 @ X + b1
  A1 = ReLU(Z1)
  Z2 = W2 @ A1 + b2
  A2 = Softmax(Z2)   <-- prediction probabilities

Backward pass (cross-entropy loss gradient through softmax simplifies nicely):
  dZ2 = A2 - Y_onehot
  dW2 = dZ2 @ A1.T / m
  db2 = mean(dZ2)
  dZ1 = W2.T @ dZ2 * ReLU'(Z1)
  dW1 = dZ1 @ X.T  / m
  db1 = mean(dZ1)
"""

import numpy as np


# ---------------------------------------------------------------------------
# Activation functions
# ---------------------------------------------------------------------------

def relu(Z):
    """Rectified Linear Unit: max(0, z)."""
    return np.maximum(0, Z)


def relu_derivative(Z):
    """Gradient of ReLU: 1 where Z > 0, else 0."""
    return (Z > 0).astype(float)


def softmax(Z):
    """
    Numerically stable Softmax.
    Subtracts the column-wise max before exponentiating to avoid overflow.
    """
    exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def cross_entropy_loss(A2, Y_onehot):
    """
    Average cross-entropy loss over the mini-batch.
      L = -1/m * sum( Y * log(A2) )
    A small epsilon is added to prevent log(0).
    """
    m = Y_onehot.shape[1]
    eps = 1e-8
    return -np.sum(Y_onehot * np.log(A2 + eps)) / m


# ---------------------------------------------------------------------------
# Parameter initialisation
# ---------------------------------------------------------------------------

def init_params(input_size=784, hidden_size=128, output_size=10, seed=42):
    """
    He-initialised weights (good default for ReLU networks).
    Biases start at zero.

    Returns a dict with keys: W1, b1, W2, b2.
    """
    rng = np.random.default_rng(seed)

    W1 = rng.standard_normal((hidden_size, input_size)) * np.sqrt(2 / input_size)
    b1 = np.zeros((hidden_size, 1))

    W2 = rng.standard_normal((output_size, hidden_size)) * np.sqrt(2 / hidden_size)
    b2 = np.zeros((output_size, 1))

    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------

def forward_pass(X, params):
    """
    Run one forward pass through the network.

    Parameters
    ----------
    X      : (784, m) — pixel values, columns are samples
    params : dict of W1, b1, W2, b2

    Returns
    -------
    A2    : (10, m) — softmax probabilities
    cache : intermediate values needed for backprop
    """
    W1, b1 = params["W1"], params["b1"]
    W2, b2 = params["W2"], params["b2"]

    Z1 = W1 @ X + b1        # (128, m)
    A1 = relu(Z1)            # (128, m)
    Z2 = W2 @ A1 + b2       # (10,  m)
    A2 = softmax(Z2)         # (10,  m)

    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
    return A2, cache


# ---------------------------------------------------------------------------
# Backward pass
# ---------------------------------------------------------------------------

def backward_pass(X, Y_onehot, params, cache):
    """
    Compute gradients via backpropagation.

    Parameters
    ----------
    X         : (784, m)
    Y_onehot  : (10,  m) — one-hot encoded labels
    params    : dict of current weights/biases
    cache     : dict from forward_pass

    Returns
    -------
    grads : dict of dW1, db1, dW2, db2
    """
    m = X.shape[1]
    W2 = params["W2"]
    Z1, A1, A2 = cache["Z1"], cache["A1"], cache["A2"]

    # Output layer gradient (softmax + cross-entropy combined derivative)
    dZ2 = A2 - Y_onehot          # (10, m)
    dW2 = dZ2 @ A1.T / m        # (10, 128)
    db2 = np.mean(dZ2, axis=1, keepdims=True)  # (10, 1)

    # Hidden layer gradient
    dA1 = W2.T @ dZ2             # (128, m)
    dZ1 = dA1 * relu_derivative(Z1)  # (128, m)
    dW1 = dZ1 @ X.T / m         # (128, 784)
    db1 = np.mean(dZ1, axis=1, keepdims=True)  # (128, 1)

    return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}


# ---------------------------------------------------------------------------
# Parameter update — SGD
# ---------------------------------------------------------------------------

def update_params(params, grads, learning_rate):
    """Vanilla gradient descent update."""
    return {
        "W1": params["W1"] - learning_rate * grads["dW1"],
        "b1": params["b1"] - learning_rate * grads["db1"],
        "W2": params["W2"] - learning_rate * grads["dW2"],
        "b2": params["b2"] - learning_rate * grads["db2"],
    }


# ---------------------------------------------------------------------------
# Parameter update — Adam
# ---------------------------------------------------------------------------

def init_adam_state(params):
    """
    Create zero-initialised first (m) and second (v) moment estimates,
    one entry per parameter tensor.
    """
    return {
        "m": {k: np.zeros_like(v) for k, v in params.items()},
        "v": {k: np.zeros_like(v) for k, v in params.items()},
        "t": 0,
    }


def adam_update(params, grads, state, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    Adam (Adaptive Moment Estimation) parameter update.

    For each parameter W with gradient dW:

      m = β1·m + (1-β1)·dW          # velocity: smoothed gradient direction
      v = β2·v + (1-β2)·dW²         # scale:    smoothed squared gradient

      m̂ = m / (1 - β1^t)            # bias-corrected (early zeros skew low)
      v̂ = v / (1 - β2^t)

      W = W - lr · m̂ / (√v̂ + ε)   # adaptive per-parameter step

    Parameters
    ----------
    params : dict  — current weights/biases (keys: W1, b1, W2, b2)
    grads  : dict  — gradients from backward_pass (keys: dW1, db1, dW2, db2)
    state  : dict  — running moments {"m": {...}, "v": {...}, "t": int}
    lr     : float — learning rate (default 0.001 works well for Adam)
    beta1  : float — momentum decay  (default 0.9)
    beta2  : float — scale decay     (default 0.999)
    eps    : float — division guard  (default 1e-8)

    Returns
    -------
    new_params : updated parameter dict
    new_state  : updated moment dict (pass back in next call)
    """
    t = state["t"] + 1
    m = state["m"]
    v = state["v"]

    new_params = {}
    new_m = {}
    new_v = {}

    for key in params:
        g = grads["d" + key]                           # e.g. "W1" → "dW1"

        new_m[key] = beta1 * m[key] + (1 - beta1) * g
        new_v[key] = beta2 * v[key] + (1 - beta2) * g ** 2

        m_hat = new_m[key] / (1 - beta1 ** t)         # bias correction
        v_hat = new_v[key] / (1 - beta2 ** t)

        new_params[key] = params[key] - lr * m_hat / (np.sqrt(v_hat) + eps)

    return new_params, {"m": new_m, "v": new_v, "t": t}


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def predict(X, params):
    """Return the predicted digit label for each column in X."""
    A2, _ = forward_pass(X, params)
    return np.argmax(A2, axis=0)


def accuracy(X, Y_labels, params):
    """Fraction of correctly classified samples."""
    preds = predict(X, params)
    return np.mean(preds == Y_labels)
