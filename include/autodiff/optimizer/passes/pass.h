#pragma once

#include "autodiff/node.h"

namespace grad::optimizer {

template<Numeric T>
class Pass {
public:
    virtual ~Pass() = default;
    virtual ExpressionPtr<T> apply_pass(ExpressionPtr<T> expression) = 0;
};

} // grad::optimizer
