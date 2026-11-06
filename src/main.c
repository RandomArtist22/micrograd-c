#define _POSIX_C_SOURCE 200809L
#include "engine.h"
#include "infer.h"
#include "nn.h"
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define N_ITERS 1000

void train_model(void) {
    srand(42);

    // Demo Dataset: 4 samples with 3 input features
    double xs[4][3] = {
        {2.0, 3.0, -1.0}, {3.0, -1.0, 0.5}, {0.5, 1.0, 1.0}, {1.0, 1.0, -1.0}};
    double ys[4] = {1.0, -1.0, -1.0, 1.0};

    // 3-layer MLP: 3 inputs -> 4 hidden -> 4 hidden -> 1 output
    size_t nouts[] = {4, 4, 1};
    MLP *model = create_mlp(3, nouts, 3);

    size_t num_params = mlp_param_count(model);
    Value **params = (Value **)malloc(sizeof(Value *) * num_params);
    mlp_get_parameters(model, params);

    printf("Number of parameters: %zu\n\n", num_params);

    // Gradient Descent Loop
    double learning_rate = 0.02;
    for (int epoch = 0; epoch < 100; epoch++) {
        mlp_zero_grad(model);

        Value *total_loss = create_value(0.0, "loss");
        for (int i = 0; i < 4; i++) {
            Value *x[3] = {create_value(xs[i][0], "x0"),
                           create_value(xs[i][1], "x1"),
                           create_value(xs[i][2], "x2")};
            Value *ypred[1];
            mlp_forward(model, x, ypred);

            Value *ygt = create_value(ys[i], "ygt");
            Value *diff = val_sub(ypred[0], ygt);
            Value *sqerr = val_mul(diff, diff);
            total_loss = val_add(total_loss, sqerr);
        }

        backward(total_loss);

        for (size_t p = 0; p < num_params; p++) {
            params[p]->data -= learning_rate * params[p]->grad;
        }

        printf("Epoch %3d | Loss: %f\n", epoch, total_loss->data);
        free_graph(total_loss);
    }

    free(params);
    free_mlp(model);
    return;
}

int check_ouptuts(double *expected, double *actual, size_t n) {
    for (size_t i = 0; i < n; i++) {
        if (fabs(expected[i] - actual[i]) > 1e-8) {
            printf("Mismatch at index %zu: expected %f, got %f\n", i,
                   expected[i], actual[i]);
            return 0;
        }
    }
    printf("Success: All outputs match expected values.\n");
    return 1;
}

int main(void) {
    // Loading the weights of the model
    InferMLP *mlp = load_weights("weights.bin");
    printf("nlayers=%zu nin =%zu\n", mlp->nlayers, mlp->nin);
    printf("layer 0, neuron 0, weight 0 : %f\n", mlp->w[0][0]);
    printf("layer 0, neuron 0, bias : %f\n", mlp->b[0][0]);

    // benchmarking test_inputs read from a binary file and verifying the
    // outputs
    size_t n_inputs, nin, n_outputs, nout;
    double *inputs = load_file_to_buffer("test_inputs.bin", &n_inputs, &nin);
    double *expected_outputs =
        load_file_to_buffer("expected_outputs.bin", &n_outputs, &nout);
    double *test_outputs = (double *)malloc(nout * n_outputs * sizeof(double));

    struct timespec t0, t1;

    // time measured for normal inference without batching
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (size_t iter = 0; iter < N_ITERS; iter++)
        for (size_t i = 0; i < n_inputs; i++)
            infer(mlp, inputs + i * mlp->nin, test_outputs + i * nout);
    clock_gettime(CLOCK_MONOTONIC, &t1);

    long total_ns =
        (t1.tv_sec - t0.tv_sec) * 1000000000L + (t1.tv_nsec - t0.tv_nsec);
    printf("Total time (normal inference): %ld ms\n", total_ns / 1000000);
    printf("Per inference (normal inference): %.1f ns\n",
           (double)total_ns / (N_ITERS * n_inputs));

    // time measured for inference with batching
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (size_t inter = 0; inter < N_ITERS; inter++) {
        infer_batch(mlp, inputs, test_outputs, n_inputs);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);

    total_ns =
        (t1.tv_sec - t0.tv_sec) * 1000000000L + (t1.tv_nsec - t0.tv_nsec);
    printf("Total time (batch inference): %ld ms\n", total_ns / 1000000);
    printf("Per inference (batch inference): %.1f ns\n",
           (double)total_ns / (N_ITERS * n_inputs));

    check_ouptuts(expected_outputs, test_outputs, nout * n_outputs);

    // time measured for blas infernece with batch size = 1
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (size_t iter = 0; iter < N_ITERS; iter++)
        for (size_t i = 0; i < n_inputs; i++)
            infer_blas(mlp, inputs + i * mlp->nin, test_outputs + i * nout);
    clock_gettime(CLOCK_MONOTONIC, &t1);

    total_ns =
        (t1.tv_sec - t0.tv_sec) * 1000000000L + (t1.tv_nsec - t0.tv_nsec);
    printf("Total time (blas inference batch_size = 1): %ld ms\n",
           total_ns / 1000000);
    printf("Per inference (blas inference batch_size = 1): %.1f ns\n",
           (double)total_ns / (N_ITERS * n_inputs));

    // time measured for blas inderence with batch size = n_inputs
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (size_t inter = 0; inter < N_ITERS; inter++) {
        infer_blas_batch(mlp, inputs, test_outputs, n_inputs);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);

    total_ns =
        (t1.tv_sec - t0.tv_sec) * 1000000000L + (t1.tv_nsec - t0.tv_nsec);
    printf("Total time (blas inference batch_size = n_inputs): %ld ms\n",
           total_ns / 1000000);
    printf("Per inference (blas inference batch_size = n_inputs): %.1f ns\n",
           (double)total_ns / (N_ITERS * n_inputs));

    check_ouptuts(expected_outputs, test_outputs, nout * n_outputs);
    return 0;
}
