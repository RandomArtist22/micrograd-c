# micrograd-c

A scalar autograd engine and neural network library in C, plus a standalone
inference engine that loads sklearn-trained models from a custom binary format
and runs them without Python, numpy, or any ML framework.

## What this is

Two things built on top of each other:

**1. Autograd engine** (`src/engine.c`, `src/nn.c`)
Reverse-mode automatic differentiation over scalar `double` values. Builds a
computation DAG dynamically, supports add/sub/mul/div/pow/neg/relu/tanh, and
runs backpropagation via topological sort. Used to train a small MLP entirely
in C with hand-written gradient descent.

**2. Inference engine** (`src/infer.c`)
A separate forward-pass-only engine with no graph, no gradients, and no
allocations in the hot path. Takes weights serialized from a trained sklearn
`MLPClassifier` and runs inference as plain double arithmetic.

The two parts share no code by design — the distinction between a training
graph and a deployment artifact is the point.

## Inference engine

### Binary weight format

Weights are serialized from Python using a custom binary format: a fixed
header describing the network shape, followed by a flat blob of IEEE 754
doubles. No JSON, no YAML — raw bytes, one `fread` per array.

```
[uint32]  nlayers
[uint32]  nin
[uint32 × nlayers]  nouts
for each layer, each neuron:
    [double × input_width]  weights
    [double]                bias
```

### Performance

Benchmarked on a 2-layer MLP (10 inputs → 64 hidden → 1 output), 1M
iterations, `-O3`, on an x86-64 Linux machine:

| Implementation | ns / inference | relative |
|---|---|---|
| This C engine | **466 ns** | 1× |
| numpy manual forward pass | 5,775 ns | 12× slower |
| sklearn `predict` | 145,116 ns | 311× slower |

The 12× gap over a raw numpy forward pass comes from Python and allocation
overhead that dominates at batch size 1. At large batch sizes, numpy's
BLAS-backed matrix multiply would pull ahead — which is the motivation for
extending this to batch inference with an explicit matrix multiply.

### Running the inference benchmark

Train and serialize a model:
```bash
python tools/save_weights.py
```

Verify C output matches Python:
```bash
python tools/verify.py
./build/micrograd
```

## Building

Prerequisites: a C11 compiler, [Meson](https://mesonbuild.com/) and [Ninja](https://ninja-build.org/).

```bash
meson setup build --buildtype=release
ninja -C build
./build/micrograd
```

## Autograd engine

Supports these operations over scalar `Value` nodes:

| Operation | Function |
|---|---|
| add, sub, mul, div | `val_add`, `val_sub`, `val_mul`, `val_div` |
| power | `val_pow` |
| negation | `val_neg` |
| ReLU | `val_relu` |
| tanh | `val_tanh` |

Memory management: `free_graph` walks the DAG and frees intermediate nodes
after each training step while leaving model parameters (`is_param = 1`) alive.
This avoids rebuilding the full graph on every epoch.

## Project structure

```
src/
  engine.c   scalar Value nodes, backward pass, topological sort
  nn.c       Neuron / Layer / MLP abstractions over Value*
  infer.c    weight loader + forward-pass-only inference engine
  main.c     training demo + inference benchmark
include/
  engine.h
  nn.h
  infer.h
tools/
  save_weights.py   train sklearn MLP, serialize to binary
  verify.py         manual numpy forward pass for numerical verification
```

## References

- Karpathy, A. [micrograd](https://github.com/karpathy/micrograd)
- Karpathy, A. [The spelled-out intro to neural networks and backpropagation](https://www.youtube.com/watch?v=VMj-3S1tku0)
