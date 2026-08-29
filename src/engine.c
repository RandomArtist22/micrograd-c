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
    strncpy(v->op, op, sizeof(v->op) - 1);
    v->op[sizeof(v->op) - 1] = '/0';
    return v; 
}

// Operation functions
Value* val_add(Value* a, Value* b) {
    Value* out = create_value(a->data + b->data, "+");
    out->prev = (Value**)malloc(sizeof(Value*) * 2);
    out->prev_count = 2;
    out->prev[0] = a;
    out->prev[1] = b;
    return out;
}

Value* val_mul(Value* a, Value* b) {
    Value* out = create_value(a->data * b->data, "*");
    out->prev = (Value**)malloc(sizeof(Value*) * 2);
    out->prev_count = 2;
    out->prev[0] = a;
    out->prev[1] = b;
    return out;
}

Value* val_relu(Value* a) {
    Value* out = create_value((a->data < 0.0) ? 0.0 : a->data, "ReLU");
    out->prev = (Value**)malloc(sizeof(Value*));
    out->prev_count = 1;
    out->prev[0] = a;
    return out;
}

Value* val_pow(Value* a, double exponent) {
    Value* out = create_value(pow(a->data, exponent), "ReLU");
    out->prev = (Value**)malloc(sizeof(Value*));
    out->prev_count = 1;
    out->prev[0] = a;
    return out;
}

