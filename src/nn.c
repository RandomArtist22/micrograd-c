#include <stdlib.h>
#include <stdio.h>
#include "nn.h"

// Random function in range [-1.0, 1.0]
static double rand_uniform(void) {
    return ((double)rand() / (double)RAND_MAX) * 2.0 - 1.0;
}

// Neuron Initialization
Neuron* create_neuron(size_t nin, bool nonlin) {
    Neuron* n = (Neuron*)malloc(sizeof(Neuron));
    n->nin = nin;
    n->nonlin = nonlin;
    n->b = create_value(0.0, "b");
    n->w = (Value**)malloc(sizeof(Value*) * nin);
    for (size_t i=0; i < nin; i++) {
        n->w[i] = create_value(rand_uniform(), "w");
    }
    return n;
}

// Neuron Forward Method
Value* neuron_forward(Neuron* n, Value** x) {
    Value* act = n->b;
    for (size_t i = 0; i < n->nin; i++) {
        Value* prod = val_mul(n->w[i], x[i]);
        act = val_add(act, prod);
    }
    return n->nonlin ? val_relu(act) : act;
}

// Freeing Neuron memory
void free_neuron(Neuron* n) {
    if (!n) return;
    free(n->w);
    free(n);
}