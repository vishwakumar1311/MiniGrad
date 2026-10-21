# MiniGrad

# Custom Micro-Autograd Engine & MLP from Scratch

This repository contains a lightweight, custom automatic differentiation engine (similar to micrograd) and a Multi-Layer Perceptron (MLP) built completely from scratch in Python, alongside PyTorch benchmarks.

## Features
- **Autograd Engine:** A scalar-value autograd engine (`Neuron` class) supporting basic mathematical operations (`+`, `-`, `*`, `/`, `**`) and activation functions (`tanh`) with dynamic backpropagation using a topologically sorted graph.
- **Graph Visualization:** Built-in helper functions (`draw_dot` using Graphviz) to trace and visualize computational graphs, showing forward pass values and backpropagated gradients.
- **Neural Network Library:** A custom modular design of Neurons, Layers, and Multi-layer Networks (`network`) capable of optimization.
- **PyTorch Comparison:** Verification and playground using official PyTorch tensors.

