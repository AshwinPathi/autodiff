#include <iostream>
#include <numbers>

#include "autodiff/optimizer/optimizer.h"
#include "autodiff/functions.h"
#include "autodiff/node.h"

template <Numeric T>
grad::ExpressionF generate_graph(const grad::ExpressionPtr<T>& x, const grad::ExpressionPtr<T>& y) {
    auto mul = x * y;

    auto expr = grad::cos(-1 * mul);

    return expr;
}

template <Numeric T>
grad::ExpressionF generate_graph2(const grad::ExpressionPtr<T>& x) {
    auto expr = x->pow(2);

    return expr;
} 


int main() {
    auto x = grad::variable<float>("x");
    auto y = grad::variable<float>("y");
    auto expr = generate_graph(x, y);

    std::unordered_map<std::string, grad::ExpressionF> values {
        // {"x", grad::constant(3.0f)},
        {"x", grad::constant(static_cast<float>(std::numbers::pi))},
        {"y", grad::constant(0.25f)},
    };

    expr->apply_variables(values);
    float result = expr->evaluate();
    std::cout << "result " << result << "\n";

    // z = cos(-(x * y))
    // x = pi
    // y = 0.25
    // dz/dx = sin(-(x * y)) * -y = -0.25 * sin(-pi/4) = 0.1767766953
    // dz/dy = sin(-(x * y)) * -x = -pi * sin(-pi/4) = 2.22144146908
    expr->get_gradients(); // --> get the gradients wrt final expr

    std::cout << x->grad() << "\n";
    std::cout << y->grad() << "\n";

    return 0;
}
