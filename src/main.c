#include <stdio.h>
#include <stdlib.h>
#include "engine.h"
#include "nn.h"

int main(void) {
    srand(42);

    // Demo Dataset: 4 samples with 3 input features
    double xs[4][3] = {
        {2.0,  3.0, -1.0},
        {3.0, -1.0,  0.5},
        {0.5,  1.0,  1.0},
        {1.0,  1.0, -1.0}
    };
    double ys[4] = {1.0, -1.0, -1.0, 1.0};

    // 3-layer MLP: 3 inputs -> 4 hidden -> 4 hidden -> 1 output
    size_t nouts[] = {4, 4, 1};
    MLP* model = create_mlp(3, nouts, 3);

    size_t num_params = mlp_param_count(model);
    Value** params = (Value**)malloc(sizeof(Value*) * num_params);
    mlp_get_parameters(model, params);

    printf("Number of parameters: %zu\n\n", num_params);

    // Gradient Descent Loop
    double learning_rate = 0.02;
    for (int epoch = 0; epoch < 100; epoch++) {
        mlp_zero_grad(model);

        Value* total_loss = create_value(0.0, "loss");
        for (int i = 0; i < 4; i++) {
            Value* x[3] = {
                create_value(xs[i][0], "x0"),
                create_value(xs[i][1], "x1"),
                create_value(xs[i][2], "x2")
            };
            Value* ypred[1];
            mlp_forward(model, x, ypred);

            Value* ygt = create_value(ys[i], "ygt");
            Value* diff = val_sub(ypred[0], ygt);
            Value* sqerr = val_mul(diff, diff);
            total_loss = val_add(total_loss, sqerr);
        }

        backward(total_loss);

        for (size_t p = 0; p < num_params; p++) {
            params[p]->data -= learning_rate * params[p]->grad;
        }

        printf("Epoch %3d | Loss: %f\n", epoch, total_loss->data);
    }

    free(params);
    free_mlp(model);
    return 0;
}