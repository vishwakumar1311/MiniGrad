# MiniGrad

A tiny scalar autograd engine and a multi-layer perceptron built on top of it, written from scratch in pure Python (no NumPy in the engine). The structure follows Andrej Karpathy's *micrograd* lecture.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vishwakumar1311/MiniGrad/blob/main/NeuralNetwork.ipynb)

## What is in the notebook

| Part | What it does |
| --- | --- |
| `Value` | Wraps one number and records how it was computed. Supports `+ - * / **`, `tanh`, and the reflected operators (`2 - x`, `1 / x`). `backward()` topologically sorts the graph and applies the chain rule in reverse. |
| `draw_dot` | Draws the computation graph with Graphviz, showing each node's value and gradient. |
| Gradient check | Builds the same expression with `Value` and with PyTorch (float64) and compares outputs and gradients. |
| `Neuron`, `Layer`, `MLP` | A `tanh` MLP whose weights are all `Value`s, so one `backward()` call gives every parameter gradient. |
| Training | A 2-sample regression problem and XOR. |

## Results

Gradient check (expression using every supported op, float64):

```
output     minigrad=0.606268543898  torch=0.606268543898
d/da       minigrad=-1.460932825690  torch=-1.460932825690  |diff|=0.00e+00
d/db       minigrad=+0.683033528894  torch=+0.683033528894  |diff|=0.00e+00
d/dc       minigrad=+0.063243845268  torch=+0.063243845268  |diff|=0.00e+00
```

The gradients also agree with central finite differences.

Training (plain gradient descent, squared-error loss):

| Problem | Network | Loss at start | Loss at end | Predictions |
| --- | --- | --- | --- | --- |
| 2-sample regression, targets 0.6 and 0.4 | 2-2-1, lr 0.05, 200 steps | 2.51 | 0.000001 | 0.6008, 0.3994 |
| XOR, targets -1 / +1 | 2-4-1, lr 0.1, 300 steps | 4.31 | 0.0064 | -0.966, 0.964, 0.956, -0.956 |

## Run it

Open the notebook in Colab with the badge above, or locally:

```bash
pip install graphviz torch jupyter     # plus the Graphviz binary for draw_dot
jupyter notebook NeuralNetwork.ipynb
```

PyTorch is only used for the gradient check. The engine itself needs just the standard library.

## Limitations

- Scalars only: every number is its own graph node, so it is far slower than tensor libraries and meant for learning, not real workloads.
- Only `tanh` as an activation, and only constant powers in `**`.
- Plain gradient descent with a fixed learning rate; no batching, optimizers or regularization.
