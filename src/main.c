#define _POSIX_C_SOURCE 200809L
#include "engine.h"
#include "infer.h"
#include "nn.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define N_ITERS 1000000

void trainModel(void) {
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

int main(void) {
    // Loading the weights of the model
    InferMLP *mlp = load_weights("weights.bin");
    printf("nlayers=%zu nin =%zu\n", mlp->nlayers, mlp->nin);
    printf("layer 0, neuron 0, weight 0 : %f\n", mlp->w[0][0][0]);
    printf("layer 0, neuron 0, bias : %f\n", mlp->b[0][0]);

    // Inferencing the model with a sample input
    double input[5][10] = {
        {-0.1032929375, 1.4515044304, 0.1618716980, -0.0309937034,
         -0.0424588555, 0.3807456699, -0.7417148536, 0.3365501791, 1.2480723311,
         -0.3174487667},
        {0.9138005982, 0.9225450001, -2.5378741891, -1.2157614914, 0.9579561369,
         0.2782144615, 0.8107124933, 0.5661881754, 1.3726476271, 0.1858732963},
        {0.6455666829, -1.7897921557, -1.5269963998, 0.6017774691, 0.5734595834,
         -1.3544575468, -0.8541012458, 0.4393457269, -1.2324040064,
         -1.1076087555},
        {2.0696062488, -0.2909335132, -1.0864807626, -0.9308452778,
         2.0688988575, 0.1316795561, 0.4399228553, 0.2805778112, 0.9618808856,
         -1.3721387382},
        {1.1362956875, -1.1886567029, -1.8514504398, -0.8092194189,
         1.0923735971, -0.4104833716, 0.5350634776, -0.0292762299,
         -0.3999200798, 0.8503742618}};
    for (int i = 0; i < 5; i++) {
        double output[1];
        infer(mlp, input[i], output);
        printf("C output: %.10f\n", output[0]);
    }

    double t_input[10];
    memcpy(t_input, input[0], sizeof(double) * 10);
    double output[1];

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    for (int i = 0; i < N_ITERS; i++) {
        infer(mlp, t_input, output);
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);

    long total_ns =
        (t1.tv_sec - t0.tv_sec) * 1000000000L + (t1.tv_nsec - t0.tv_nsec);
    printf("Total: %ld ms\n", total_ns / 1000000);
    printf("Per inference: %.1f ns\n", (double)total_ns / N_ITERS);
    return 0;
}
