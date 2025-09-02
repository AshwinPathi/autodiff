#pragma once

#include "autodiff/node.h"
#include "autodiff/optimizer/passes/pass.h"

namespace grad::optimizer {

template<Numeric T>
ExpressionPtr<T> optimize(ExpressionPtr<T> input_graph, const std::vector<std::shared_ptr<Pass<T>>>& passes) {
    // Need a real deep copy here, don't have that right now.
    ExpressionPtr<T> new_graph = input_graph;
    for (const auto& pass : passes) {
        new_graph = pass->apply_pass(new_graph);
    }
    return new_graph;
}

} // grad::optimizer
