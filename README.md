# micrograd-c

A minimal scalar autograd engine and neural network implementation in C, inspired by Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd).

## Features

- **Scalar Autograd Engine**: Dynamically builds a directed acyclic graph (DAG) and computes gradients via reverse-mode automatic differentiation.
- **Supported Operations**: Addition (`+`), subtraction (`-`), multiplication (`*`), division (`/`), power (`pow`), negation, ReLU, and tanh.
- **Neural Network Library**: Modular `Neuron`, `Layer`, and `MLP` abstractions.
- **Memory Management**: Graph traversal and cleanup (`free_graph`) that reclaims intermediate nodes after each training step while preserving model parameters.

## Building and Running

Prerequisites:
- A C11 compiler (GCC or Clang)
- [Meson](https://mesonbuild.com/) and [Ninja](https://ninja-build.com/)

```bash
# Setup build directory
meson setup build

# Compile
ninja -C build

# Run the demo
./build/micrograd
```

The included `main.c` trains a 3-layer MLP on a toy classification dataset using gradient descent.
