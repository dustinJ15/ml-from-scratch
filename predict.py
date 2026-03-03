"""
Quick inference demo — loads saved weights and predicts a random test image.

Run:
    python predict.py
    python predict.py --index 42   # predict a specific sample
"""

import argparse
import numpy as np
from neural_network import forward_pass
from utils import load_mnist


def show_digit(pixels_784):
    """Render a 28x28 digit in the terminal using ASCII art."""
    grid = pixels_784.reshape(28, 28)
    chars = " .:-=+*#%@"
    for row in grid:
        print("".join(chars[int(v * (len(chars) - 1))] for v in row))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=None,
                        help="Index of test sample to predict (random if omitted)")
    args = parser.parse_args()

    # Load saved parameters
    try:
        data = np.load("trained_params.npz")
    except FileNotFoundError:
        print("No trained_params.npz found. Run train.py first.")
        return

    params = {k: data[k] for k in data.files}

    # Load test set
    _, X_test, _, Y_test = load_mnist()

    # Pick sample
    idx = args.index if args.index is not None else np.random.randint(X_test.shape[1])
    x = X_test[:, idx : idx + 1]   # (784, 1)
    true_label = Y_test[idx]

    # Predict
    A2, _ = forward_pass(x, params)
    predicted = int(np.argmax(A2, axis=0)[0])
    confidence = float(A2[predicted, 0]) * 100

    print(f"\nSample index : {idx}")
    print(f"True label   : {true_label}")
    print(f"Predicted    : {predicted}  (confidence: {confidence:.1f}%)\n")

    show_digit(x[:, 0])

    print("\nClass probabilities:")
    for digit, prob in enumerate(A2[:, 0]):
        bar = "█" * int(prob * 40)
        print(f"  {digit}: {bar:<40} {prob * 100:5.1f}%")


if __name__ == "__main__":
    main()
