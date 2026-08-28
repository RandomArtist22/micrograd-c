#ifndef ENGINE_H
#define ENGINE_H

typedef struct Value {
  double data;
  double grad;
  void (*backward)(struct Value *);
  struct Value **prev;
  char op[4];
} Value;

Value *create_value(double data);
Value *val_add(Value *a, Value *b);
Value *val_mul(Value *a, Value *b);
Value *val_tanh(Value *a);
void backward(Value *root);
void free_value(Value *v);

#endif // !ENGINE_H
