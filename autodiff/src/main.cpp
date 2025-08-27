#include <iostream>
#include <numbers>

#include "autodiff/functions.h"
#include "autodiff/node.h"

grad::ExpressionF sigmoid(grad::ExpressionF x) {
    return 1.f / (1 + grad::exp(-1 * x));
}

int main() {
    auto x = grad::constant(1.f);
    auto sigm = sigmoid(x);

    sigm->get_gradients()();
    std::cout << sigm->value() << "\n";
    std::cout << x->grad() << "\n"; // ds/dx @ x = 1

    return 0;
}
