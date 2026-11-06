#include "infer.h"
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

InferMLP *load_weights(const char *path) {
    InferMLP *mlp = (InferMLP *)malloc(sizeof(InferMLP));
    FILE *fp = fopen(path, "rb");
    if (fp == NULL) {
        perror("Failed to open file");
        exit(1);
    }

    fread(&mlp->nlayers, sizeof(uint32_t), 1, fp);
    fread(&mlp->nin, sizeof(uint32_t), 1, fp);

    mlp->nouts = (size_t *)malloc(sizeof(size_t) * mlp->nlayers);
    for (size_t i = 0; i < mlp->nlayers; i++) {
        uint32_t temp;
        fread(&temp, sizeof(uint32_t), 1, fp);
        mlp->nouts[i] = (size_t)temp;
    }

    mlp->w = (double ***)malloc(mlp->nlayers * sizeof(double **));
    mlp->b = (double **)malloc(mlp->nlayers * sizeof(double *));
    for (size_t i = 0; i < mlp->nlayers; i++) {
        mlp->b[i] = (double *)malloc(mlp->nouts[i] * sizeof(double *));
        mlp->w[i] = (double **)malloc(mlp->nouts[i] * sizeof(double *));
        for (size_t j = 0; j < mlp->nouts[i]; j++) {
            size_t input_size = (i == 0) ? mlp->nin : mlp->nouts[i - 1];
            mlp->w[i][j] = (double *)malloc(input_size * sizeof(double));
            fread(mlp->w[i][j], sizeof(double), input_size, fp);
            fread(&mlp->b[i][j], sizeof(double), 1, fp);
        }
    }
    fclose(fp);
    return mlp;
}

double *load_file_to_buffer(const char *path, size_t *n, size_t *ndim) {
    FILE *fp = fopen(path, "rb");
    if (fp == NULL) {
        perror("Failed to open file");
        exit(1);
    }
    fread(n, sizeof(uint32_t), 1, fp);
    fread(ndim, sizeof(uint32_t), 1, fp);

    double *inputs = (double *)malloc((*n) * (*ndim) * sizeof(double));
    fread(inputs, sizeof(double), (*n) * (*ndim), fp);

    return inputs;
}

void infer(InferMLP *m, const double *input, double *output) {
    double *curr = (double *)malloc(m->nin * sizeof(double));
    memcpy(curr, input, m->nin * sizeof(double));
    for (size_t i = 0; i < m->nlayers; i++) {
        double *next = (double *)malloc(m->nouts[i] * sizeof(double));
        for (size_t j = 0; j < m->nouts[i]; j++) {
            next[j] = m->b[i][j];
            size_t input_size = (i == 0) ? m->nin : m->nouts[i - 1];
            for (size_t k = 0; k < input_size; k++) {
                next[j] += m->w[i][j][k] * curr[k];
            }
            if (i != m->nlayers - 1)
                next[j] = (next[j] > 0) ? next[j] : 0;
        }
        free(curr);
        curr = next;
    }
    memcpy(output, curr, m->nouts[m->nlayers - 1] * sizeof(double));
    free(curr);
}

void infer_batch(InferMLP *m, const double *inputs, double *outputs, size_t n) {
    size_t nout = m->nouts[m->nlayers - 1];
    for (size_t i = 0; i < n; i++) {
        infer(m, inputs + i * m->nin, outputs + i * nout);
    }
}
