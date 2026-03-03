"""
Data utilities for loading and preprocessing MNIST.

MNIST is a dataset of 70,000 grayscale images of handwritten digits (0-9).
Each image is 28x28 pixels = 784 values, each in [0, 255].

Downloads the four original binary files from a reliable mirror and parses
them according to the IDX file format spec (http://yann.lecun.com/exdb/mnist/).
Files are cached in ~/.cache/mnist_scratch/ after the first download.
"""

import gzip
import os
import struct
import urllib.request
from pathlib import Path

import numpy as np  # only non-stdlib dependency

# PyTorch's S3 mirror — stable and fast
_MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist/"
_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images":  "t10k-images-idx3-ubyte.gz",
    "test_labels":  "t10k-labels-idx1-ubyte.gz",
}
_CACHE_DIR = Path.home() / ".cache" / "mnist_scratch"


def _download(filename: str) -> Path:
    """Download a gzipped MNIST file to the cache directory if not present."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dest = _CACHE_DIR / filename
    if not dest.exists():
        url = _MIRROR + filename
        print(f"  Downloading {filename} ...", end=" ", flush=True)
        urllib.request.urlretrieve(url, dest)
        print("done")
    return dest


def _parse_images(path: Path) -> np.ndarray:
    """Parse an IDX3 image file → float64 array shaped (n, 784) in [0, 1]."""
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, f"Bad magic number: {magic}"
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n, rows * cols).astype(np.float64) / 255.0


def _parse_labels(path: Path) -> np.ndarray:
    """Parse an IDX1 label file → int array shaped (n,)."""
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        assert magic == 2049, f"Bad magic number: {magic}"
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.astype(int)


def load_mnist():
    """
    Load MNIST, downloading from the PyTorch S3 mirror on the first run.
    Files are cached in ~/.cache/mnist_scratch/ and reused on subsequent runs.

    Returns
    -------
    X_train : (784, 60000)  float64 in [0, 1]
    X_test  : (784, 10000)  float64 in [0, 1]
    Y_train : (60000,)      int labels 0-9
    Y_test  : (10000,)      int labels 0-9
    """
    print("Loading MNIST... (downloads ~12 MB on first run, cached after)")
    X_train = _parse_images(_download(_FILES["train_images"]))
    Y_train = _parse_labels(_download(_FILES["train_labels"]))
    X_test  = _parse_images(_download(_FILES["test_images"]))
    Y_test  = _parse_labels(_download(_FILES["test_labels"]))

    # Transpose so columns = samples (convention used throughout this project)
    return X_train.T, X_test.T, Y_train, Y_test


def one_hot_encode(Y, num_classes=10):
    """
    Convert an integer label vector into a one-hot matrix.

    Example: label 3 → [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]

    Parameters
    ----------
    Y           : (m,)  integer labels
    num_classes : number of output classes

    Returns
    -------
    (num_classes, m) float matrix
    """
    m = Y.shape[0]
    one_hot = np.zeros((num_classes, m))
    one_hot[Y, np.arange(m)] = 1.0
    return one_hot


def get_mini_batches(X, Y_onehot, batch_size, rng=None):
    """
    Yield (X_batch, Y_batch) tuples for each mini-batch.

    Parameters
    ----------
    X         : (784, m)
    Y_onehot  : (10,  m)
    batch_size : int
    rng        : optional numpy random Generator for shuffling
    """
    m = X.shape[1]
    indices = np.arange(m)
    if rng is not None:
        rng.shuffle(indices)

    for start in range(0, m, batch_size):
        idx = indices[start: start + batch_size]
        yield X[:, idx], Y_onehot[:, idx]
