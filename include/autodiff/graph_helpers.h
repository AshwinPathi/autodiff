#pragma once

#include <deque>

namespace grad::graph {

enum class TraversalType {
    DFS,
    BFS
};

/**
BFS or BFS traversal.
- Template Args:
    - TType: TraversalType::DFS or TraversalType::BFS
- Function Args:
    - start: Starting node
    - get_children_func: Function that takes in a node and returns its children as an iterable
    - apply_func: Function that takes in a node and applies some operation on it (can be empty).
*/
template <TraversalType TType, typename NodeType, typename GetChildrenFunc, typename Applyfunc>
requires requires (NodeType x, GetChildrenFunc gcf, Applyfunc af) {
    // Require the output to be some iterable range (probably a better way of doing this)
    {gcf(x).begin()}; {gcf(x).end()};
    // Require the application function to at least take the input node as an argument
    // (though it can do nothing with it)
    {af(x)};
}
void traverse(const NodeType& start, GetChildrenFunc get_children_func, Applyfunc apply_func) {
    std::unordered_set<NodeType> visited;
    std::deque<NodeType> frontier;

    auto add_to_frontier = [&](const NodeType& node) -> void {
        frontier.push_back(node);
    };
    
    auto remove_from_frontier = [&]() -> NodeType {
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
        apply_func(node);
        for (const auto& input : get_children_func(node)) {
            if (visited.find(input) == visited.end()) {
                add_to_frontier(input);
            }
        }
    }
}

} // grad::graph
