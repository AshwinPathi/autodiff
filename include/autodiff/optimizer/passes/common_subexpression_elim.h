#pragma once

#include "autodiff/node.h"

namespace grad::optimizer {

template<Numeric T>
class CommonSubexpressionElimPass : public Pass<T> {
public:
    ~CommonSubexpressionElimPass() override = default;

    ExpressionPtr<T> apply_pass(ExpressionPtr<T> expression) override {
        // TODO
        return expression;
    }
};

} // grad::optimizer
