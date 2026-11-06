#ifndef INFER_H
#define INFER_H

#include <stddef.h>

typedef struct {
    size_t nlayers;
    size_t nin;    // array of length nlayers
    size_t *nouts; // w[layer][neuron][weight_index]
    double ***w;   // b[layer][neuron]
    double **b;
} InferMLP;

InferMLP *load_weights(const char *path);
void infer(InferMLP *mlp, const double *input, double *output);

#endif
