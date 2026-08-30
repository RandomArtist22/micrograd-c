#include "engine.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_NODES 4096

// Node allocation function
Value* create_value(double data, const char* op) {
    Value* v = (Value*)malloc(sizeof(Value));
    v->data = data;
    v->grad = 0.0;
    v->backward = NULL;
    v->prev = NULL;
    v->prev_count = 0;
    v->visited = 0;
    if (op != NULL) {
        strncpy(v->op, op, sizeof(v->op) - 1);
        v->op[sizeof(v->op) - 1] = '\0';
    } else {
        v->op[0] = '\0';
    }
    return v; 
}

// Internal backward calls
static void backward_add(Value* out) {
    out->prev[0]->grad += 1.0 * out->grad;
    out->prev[1]->grad += 1.0 * out->grad;
}

static void backward_mul(Value* out) {
    out->prev[0]->grad += out->prev[1]->data * out->grad;
    out->prev[1]->grad += out->prev[0]->data * out->grad;
}

static void backward_relu(Value* out) {
    out->prev[0]->grad += (out->data > 0.0 ? 1.0 : 0.0) * out->grad;
}

static void backward_pow(Value *out) {
    Value* base = out->prev[0];
    double n = out->prev[1]->data;

    base->grad += (n * pow(base->data, n - 1.0)) * out->grad;
}

static void backward_tanh(Value* out) {
    out->prev[0]->grad += (1 - out->data * out->data) * out->grad;
}

// Operation functions
Value* val_add(Value* a, Value* b) {
    Value* out = create_value(a->data + b->data, "+");
    out->prev = (Value**)malloc(sizeof(Value*) * 2);
    out->prev_count = 2;
    out->prev[0] = a;
    out->prev[1] = b;
    out->backward = backward_add;
    return out;
}

Value* val_mul(Value* a, Value* b) {
    Value* out = create_value(a->data * b->data, "*");
    out->prev = (Value**)malloc(sizeof(Value*) * 2);
    out->prev_count = 2;
    out->prev[0] = a;
    out->prev[1] = b;
    out->backward = backward_mul;
    return out;
}

Value* val_relu(Value* a) {
    Value* out = create_value((a->data < 0.0) ? 0.0 : a->data, "ReLU");
    out->prev = (Value**)malloc(sizeof(Value*));
    out->prev_count = 1;
    out->prev[0] = a;
    out->backward = backward_relu;
    return out;
}

Value* val_tanh(Value* a) {
    Value* out = create_value(tanh(a->data), "tanh");
    out->prev = (Value**)malloc(sizeof(Value*));
    out->prev_count = 1;
    out->prev[0] = a;
    out->backward = backward_tanh;
    return out;
}

Value* val_pow(Value* a, double exponent) {
    Value* exp_node = create_value(exponent, "const");

    Value* out = create_value(pow(a->data, exponent), "ReLU");
    out->prev = (Value**)malloc(sizeof(Value*));
    out->prev_count = 1;
    out->prev[0] = a;
    out->prev[1] = exp_node;
    out->backward = backward_pow;
    return out;
}

Value* val_neg(Value* a) {
    Value* neg_one = create_value(-1.0, NULL);
    return val_mul(a, neg_one);
}

Value* val_sub(Value* a, Value* b) {
    return val_add(a, val_neg(b));
}

Value* val_div(Value* a, Value* b) {
    return val_mul(a, val_pow(b, -1.0));
}

// Topological Sorting
static void build_topo(Value* v, Value** topo_list, size_t* topo_size) {
    if (!v || v->visited) return;
    v->visited = 1;

    for(int i = 0; i < (int)v->prev_count; i++) {
        build_topo(v->prev[i], topo_list, topo_size);
    }

    topo_list[(*topo_size)++] = v;
}

// Backward Function for Graph
void backward(Value* root) {
    Value* topo[MAX_NODES];
    size_t topo_size = 0;

    build_topo(root, topo, &topo_size);

    root->grad = 1.0;
    for(int i = (int)topo_size - 1; i >= 0; i--) {
        if (topo[i]->backward) {
            topo[i]->backward(topo[i]);
        }
    }
}

// Free Memory of Graph
void free_graph(Value* root) {
    Value* topo[MAX_NODES];
    size_t topo_size = 0;

    build_topo(root, topo, &topo_size);

    for(int i = (int)topo_size - 1; i >= 0; i--) {
        if (topo[i]->prev) free(topo[i]->prev);
        free(topo[i]); 
    } 
}





