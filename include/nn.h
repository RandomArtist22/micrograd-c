#ifndef NN_H
#define NN_H

#include <stddef.h>
#include <stdbool.h>
#include "engine.h"

// Structure Definitions
typedef struct {
    Value** w;
    size_t nin;
    Value* b;
    bool nonlin;
} Neuron;

// Neuron API
Neuron* create_neuron(size_t nin, bool nonlin);
Value* neuron_forward(Neuron* n, Value** x);
void free_neuron(Neuron* n);

#endif NN_H