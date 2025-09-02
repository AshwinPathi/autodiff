#pragma once

#include <deque>
#include "autodiff/node.h"

namespace grad::graph {

enum class TraversalType {
    DFS,
    BFS
};

template <TraversalType TType, Numeric T, typename Func>
void traverse(const ExpressionPtr<T>& start, Func func = {}) {
    std::unordered_set<ExpressionPtr<T>> visited;
    std::deque<ExpressionPtr<T>> frontier;

    auto add_to_frontier = [&](const ExpressionPtr<T>& node) -> void {
        frontier.push_back(node);
    };
    
    auto remove_from_frontier = [&]() -> ExpressionPtr<T> {
        if constexpr (TType == TraversalType::DFS) {
            auto node = frontier.back();
            frontier.pop_back();
            return node;
        } else {
            auto node = frontier.front();
            frontier.pop_front();
            return node;
        }
    };

    add_to_frontier(start);

    while (!frontier.empty()) {
        auto node = remove_from_frontier();
        if (visited.find(node) != visited.end()) {
            continue;
        }
        visited.insert(node);
        func(node);
        for (const auto& input : node->get_inputs()) {
            if (visited.find(input) == visited.end()) {
                add_to_frontier(input);
            }
        }
    }
}

} // grad::graph
