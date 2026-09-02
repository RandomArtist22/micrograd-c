#ifndef ENGINE_H
#define ENGINE_H

#include <stddef.h>

// Computational graph node structure
typedef struct Value {
  double data;
  double grad;
  void (*backward)(struct Value *);
  struct Value **prev;
  size_t prev_count;
  char op[8];
  int visited;
  int is_param;
} Value;

// Function Prototypes for the engine
Value* create_value(double data, const char* op);
Value* val_add(Value* a, Value* b);
Value* val_mul(Value* a, Value* b);
Value* val_pow(Value* a, double exponent);
Value* val_relu(Value* a);
Value* val_neg(Value* a);
Value* val_sub(Value* a, Value* b);
Value* val_div(Value* a, Value* b);
Value* val_tanh(Value* a);
void backward(Value* root);
void free_graph(Value* root);


#endif // !ENGINE_H
