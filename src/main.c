#include "engine.h"
#include <stdio.h>

int main(void) {
    Value* a = create_value(2.0, "a");
    Value* b = create_value(-3.0, "b");
    Value* c = create_value(10.0, "c");

    Value* e = val_mul(a, b);  
    Value* L = val_add(e, c); 

    backward(L);

    printf("Forward Pass: L = %f (Expected: 4.0)\n", L->data);
    printf("dL/da = %f (Expected: -3.0)\n", a->grad);
    printf("dL/db = %f (Expected: 2.0)\n", b->grad);
    printf("dL/dc = %f (Expected: 1.0)\n", c->grad);

    free_graph(L);
    return 0;
}
