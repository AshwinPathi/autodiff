#pragma once

#include "autodiff/node.h"

namespace grad::optimizer {

template<Numeric T>
class ConstantFolding : public Pass<T> {
public:
    ExpressionPtr<T> operator()(const ExpressionPtr<T>& expression) override {
        ExpressionPtr<T> new_expression;
    }
private:
    
};

} // grad::optimizer
