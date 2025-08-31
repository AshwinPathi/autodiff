#pragma once

#include <unordered_set>

#include "autodiff/node.h"

namespace grad::optimizer {

template<Numeric T>
class DeadCodeElimPass : public Pass<T> {
public:
    ~DeadCodeElimPass() override = default;

    ExpressionPtr<T> apply_pass(const ExpressionPtr<T>& expression) override {
        std::unordered_set<ExpressionPtr<T>> reachable;

        auto mark_reachable_nodes = [&](const ExpressionPtr<T>& node) {
            // TODO - make not recursive
            if (reachable.find(node) != reachable.end()) {
                return;
            }
            reachable.insert(node);
            for (const auto& input : node->get_inputs()) {
                mark_reachable_nodes(input);
            }
        };

        mark_reachable_nodes(expression);
        return expression;
    }
};

} // grad::optimizer
