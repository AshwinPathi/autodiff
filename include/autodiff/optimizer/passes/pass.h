#pragma once

#include "autodiff/node.h"

namespace grad::optimizer {

template<Numeric T>
class Pass {
public:
    virtual ExpressionPtr<T> operator()(const ExpressionPtr<T>& expression) = 0;
};

} // grad::optimizer
