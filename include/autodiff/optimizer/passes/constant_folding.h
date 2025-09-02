#pragma once

#include <iostream>

#include <algorithm>

#include "autodiff/node.h"
#include "autodiff/optimizer/graph_helpers.h"

namespace grad::optimizer {

template<Numeric T>
class ConstantFoldingPass : public Pass<T> {
public:
    ~ConstantFoldingPass() override = default;

    ExpressionPtr<T> apply_pass(ExpressionPtr<T> expression) override {
        // Can apply this pass in two phases:
        // 1st phase - marking:
        //      In this phase, just mark nodes as constant if they are constant or if all
        //      their inputs are provably constant.
        // 2nd phase - folding: In this phase, actually fold the constants by evaluating the constant nodes.
        mark_as_const_pass(expression);
        fold_constants(expression);
        return expression;
    }

private:
    void mark_as_const_pass(ExpressionPtr<T> expression) {
        auto helper = [&](this const auto &self, ExpressionPtr<T> expression) -> bool {
            if (expression->get_op() == Op::CONSTANT) {
                return true;
            }

            bool all_const_inputs = expression->get_inputs().size() > 0;
            for (const auto& input : expression->get_inputs()) {
                all_const_inputs &= self(input);
            }

            if (all_const_inputs) {
                expression->mark_as_const();
                return true;
            }

            return false;

        }(expression);
    }

    void fold_constants(ExpressionPtr<T> expression) {
        graph::traverse<graph::TraversalType::DFS>(expression, [](ExpressionPtr<T> node) {
            if (node->get_op() == Op::CONSTANT) {
                // TODO - Cache evaluations so we don't keep recomputing this value.
                // TODO - have some cost function so that we don't fold everything
                node->evaluate();
                // Inputs are no longer necessary if we have proven that this node is constant
                // and have evaluated it (Other nodes that depend on the same inputs will still persist)
                node->clear_inputs();
            }
        });
    }
};

} // grad::optimizer
