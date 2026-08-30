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

typedef struct {
    Neuron* neurons;
    size_t nin;
    size_t nout;
} Layer;

typedef struct {
    Layer* layers;
    size_t nlayers;
} MLP;

// Neuron API
Neuron* create_neuron(size_t nin, bool nonlin);
Value* neuron_forward(Neuron* n, Value** x);
void free_neuron(Neuron* n);

// Layer API
Layer* create_layer(size_t nin, size_t nout, bool nonlin);
void layer_forward(Layer* l, Value** x, Value** out);
void free_layer(Layer* l);

// MLP API
MLP* create_mlp(size_t nin, const size_t* nouts, size_t nlayers);
void mlp_forward(MLP* mlp, Value** x, Value** out);
void free_mlp(MLP* mlp);

// Parameter Helper Functions
size_t mlp_param_count(MLP* mlp);
void mlp_get_parameters(MLP* mlp, Value** params);
void mlp_zero_grad(MLP* mlp);

#endif