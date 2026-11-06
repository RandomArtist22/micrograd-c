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

Benchmarked on a 2-layer MLP (10 inputs → 64 hidden → 1 output), 1K
iterations, `-O3`, x86-64 Linux, n=6000:

| Implementation | ns / inference | relative |
|---|---|---|
| BLAS batch (`cblas_dgemm`) | **111 ns** | 1× |
| Scalar loop (single) | 295 ns | 2.6× slower |
| BLAS single | 507 ns | 4.6× slower |
| numpy manual forward pass | 5,775 ns | 52× slower |
| sklearn `predict` | 145,116 ns | 1307× slower |

BLAS batch wins at large n because a single `cblas_dgemm` call processes
the entire batch as one matrix multiply, letting OpenBLAS use AVX vectorization
across all inputs at once. BLAS single is slower than the scalar loop — the
OpenBLAS dispatch overhead isn't worth it for a 64×1 multiply.

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

Prerequisites: a C11 compiler, [Meson](https://mesonbuild.com/), [Ninja](https://ninja-build.org/), and OpenBLAS.

```bash
# Arch
sudo pacman -S openblas

PKG_CONFIG_PATH=/usr/lib/pkgconfig meson setup build --buildtype=release
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
