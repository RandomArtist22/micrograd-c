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
        n->w[i]->is_param = 1;
    }
    n->b->is_param = 1;
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
    for (size_t i = 0; i < n->nin; i++) {
        free(n->w[i]);
    }
    free(n->w);
    free(n->b);
    free(n);
}

// Layer Initialization
Layer* create_layer(size_t nin, size_t nout, bool nonlin) {
    Layer* l = (Layer*)malloc(sizeof(Layer));
    l->nin = nin;
    l->nout = nout;
    l->neurons = (Neuron*)malloc(sizeof(Neuron) * nout);
    for (size_t i = 0; i < nout; i++) {
        Neuron* n = create_neuron(nin, nonlin);
        l->neurons[i] = *n;
        free(n);
    }
    return l;
}

// Layer Forward Method
void layer_forward(Layer* l, Value** x, Value** out) {
    for (size_t i = 0; i < l->nout; i++) {
        out[i] = neuron_forward(&l->neurons[i], x);
    }
}

// Freeing Layer memory
void free_layer(Layer* l) {
    if (!l) return;
    for (size_t i = 0; i < l->nout; i++) {
        free(l->neurons[i].w);
    }
    free(l->neurons);
    free(l);
}

// MLP Initialization
MLP* create_mlp(size_t nin, const size_t* nouts, size_t nlayers) {
    MLP* mlp = (MLP*)malloc(sizeof(MLP));
    mlp->nlayers = nlayers;
    mlp->layers = (Layer*)malloc(sizeof(Layer) * nlayers);

    size_t curr_in = nin;
    for (size_t i = 0; i < nlayers; i++) {
        bool nonlin = (i != nlayers - 1);
        Layer* l = create_layer(curr_in, nouts[i], nonlin);
        mlp->layers[i] = *l;
        free(l);
        curr_in = nouts[i];
    }
    return mlp;
}

// MLP Forward Method
void mlp_forward(MLP* mlp, Value** x, Value** out) {
    Value** curr_in = x;
    Value** layer_out = NULL;

    for (size_t i = 0; i < mlp->nlayers; i++) {
        layer_out = (Value**)malloc(sizeof(Value*) * mlp->layers[i].nout);
        layer_forward(&mlp->layers[i], curr_in, layer_out);

        if (i > 0) free(curr_in);
        curr_in = layer_out;
    }

    for (size_t i = 0; i < mlp->layers[mlp->nlayers - 1].nout; i++) {
        out[i] = layer_out[i];
    }
    free(layer_out);
}

// Freeing MLP memory
void free_mlp(MLP* mlp) {
    if (!mlp) return;
    for (size_t i = 0; i < mlp->nlayers; i++) {
        for (size_t j = 0; j < mlp->layers[i].nout; j++) {
            for (size_t k = 0; k < mlp->layers[i].neurons[j].nin; k++) {
                free(mlp->layers[i].neurons[j].w[k]);
            }
            free(mlp->layers[i].neurons[j].w);
            free(mlp->layers[i].neurons[j].b);
        }
        free(mlp->layers[i].neurons);
    }
    free(mlp->layers);
    free(mlp);
}

// Getting total Parameter count
size_t mlp_param_count(MLP* mlp) {
    size_t total = 0;
    for (size_t i = 0; i < mlp->nlayers; i++) {
        total += mlp->layers[i].nout * (mlp->layers[i].nin + 1);
    }
    return total;
}

// Storing all Parameter values
void mlp_get_parameters(MLP* mlp, Value** params) {
    size_t indx = 0;
    for (size_t i = 0; i < mlp->nlayers; i++) {
        for (size_t j = 0; j < mlp->layers[i].nout; j++) {
            for (size_t k = 0; k < mlp->layers[i].neurons[j].nin; k++) {
                params[indx++] = mlp->layers[i].neurons[j].w[k];
            }
            params[indx++] = mlp->layers[i].neurons[j].b;
        }
    }
}

// Reseting gradients after backward pass
void mlp_zero_grad(MLP* mlp) {
    size_t total = mlp_param_count(mlp);
    Value** params = (Value**)malloc(sizeof(Value*) * total);
    mlp_get_parameters(mlp, params);
    for (size_t i = 0; i < total; i++) {
        params[i]->grad = 0.0;
    }
    free(params);
}
