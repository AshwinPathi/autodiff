#include <iostream>
#include <numbers>

#include "autodiff/functions.h"
#include "autodiff/node.h"

grad::ExpressionF sigmoid(grad::ExpressionF x) {
    return 1.f / (1 + grad::exp(-1 * x));
}

grad::ExpressionF random_pow(grad::ExpressionF x) {
    return grad::pow(2.f, x);
}

int main() {
    auto x = grad::constant(1.f);
    auto sigm = sigmoid(x);

    sigm->get_gradients();
    std::cout << sigm->value() << "\n";
    std::cout << x->grad() << "\n"; // ds/dx @ x = 1


    auto y = grad::constant(2.f);
    auto p = random_pow(y);
    p->get_gradients();
    std::cout << p->value() << "\n";
    std::cout << y->grad() << "\n"; // ds/dx @ y = 1

    return 0;
}
