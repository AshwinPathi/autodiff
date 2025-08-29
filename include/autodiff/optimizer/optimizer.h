#pragma once

#include "autodiff/node.h"
#include "autodiff/optimizer/passes/pass.h"

namespace grad::optimizer {

template<Numeric T>
ExpressionPtr<T> get_new_graph(const ExpressionPtr<T>& input_graph, const std::vector<Pass<T>>& passes) {
    ExpressionPtr<T> new_graph = input_graph;
    for (const auto& pass : passes) {
        new_graph = pass(new_graph);
    }
    return new_graph;
}

} // grad::optimizer
