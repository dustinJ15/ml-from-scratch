# Neural Network from Scratch

A fully-connected neural network trained on MNIST, built using only NumPy — no PyTorch, no TensorFlow, no ML frameworks of any kind.

The goal was to implement every piece by hand to understand what's actually happening inside a neural network: how gradients flow backward, why certain optimizers converge faster, and what the learned weights look like visually.

---

## Architecture

```
Input (784) → Hidden (128, ReLU) → Output (10, Softmax)
```

- **784** inputs — one per pixel in a 28×28 grayscale image
- **ReLU** hidden layer — fires when the weighted sum is positive, silent otherwise
- **Softmax** output — converts raw scores into a probability distribution over digits 0–9
- **Cross-entropy loss** — penalizes confident wrong predictions heavily
- **He initialization** — scales initial weights to keep signal stable through ReLU layers

---

## What's Implemented from Scratch

| Component | Details |
|---|---|
| Forward pass | Matrix multiply → activation, repeated per layer |
| Backpropagation | Chain rule gradient computation, layer by layer |
| SGD | Vanilla mini-batch gradient descent |
| Adam | Adaptive per-parameter learning rates with bias correction |
| Cross-entropy loss | With numerical stability (log clipping) |
| Softmax | Numerically stable (max subtraction before exp) |
| He initialization | Correct variance scaling for ReLU networks |
| MNIST loader | Downloads and parses the raw IDX binary format directly |

---

## Results

| Optimizer | Epochs | Test Accuracy |
|---|---|---|
| SGD (lr=0.1) | 20 | ~97.5% |
| Adam (lr=0.001) | 20 | ~98.0% |

---

## Getting Started

```bash
pip install numpy
python train.py        # trains the network, saves weights to trained_params.npz
python predict.py      # loads weights, predicts a random test digit
```

`predict.py` renders the digit in ASCII and shows a probability bar for each class:

```
Sample index : 4821
True label   : 3
Predicted    : 3  (confidence: 99.1%)

   .:-=+*#%@
  ...

Class probabilities:
  0:                                          0.0%
  3: ████████████████████████████████████    99.1%
  ...
```

---

## Optimizers

Switch between SGD and Adam in `train.py`:

```python
optimizer = "adam"   # or "sgd"
```

Adam tracks a smoothed gradient direction (momentum) and a per-parameter scale based on recent gradient magnitudes. Parameters that receive large, consistent gradients get smaller steps; sparse or uncertain gradients get larger steps. This removes the need to carefully tune a single global learning rate.

---

## Project Structure

```
├── neural_network.py   # forward pass, backprop, SGD, Adam, activations, loss
├── train.py            # training loop
├── utils.py            # MNIST download/parse, one-hot encoding, mini-batch generator
├── predict.py          # inference demo with ASCII visualization
└── requirements.txt
```

---

## References

- Sassoli, B. (2022). [How to build a neural network from zero](https://towardsdatascience.com/building-a-neural-network-from-scratch-8f03c5c50adc/). *Towards Data Science*.
- Kingma, D. P., & Ba, J. (2014). [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980). *arXiv*.
- LeCun, Y. et al. MNIST dataset. http://yann.lecun.com/exdb/mnist/
