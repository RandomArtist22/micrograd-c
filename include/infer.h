#ifndef INFER_H
#define INFER_H

#include <stddef.h>

typedef struct {
    size_t nlayers;
    size_t nin; // array of length nlayers
    size_t *nouts;
    double ***w; // w[layer][neuron][input]
    double **b;  // b[layer][neuron]
} InferMLP;

InferMLP *load_weights(const char *path);
void infer(InferMLP *mlp, const double *input, double *output);
void infer_batch(InferMLP *mlp, const double *input, double *output, size_t n);
double *load_file_to_buffer(const char *path, size_t *n, size_t *ndim);

#endif
