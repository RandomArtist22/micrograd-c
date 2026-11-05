#include "infer.h"
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

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
