#pragma once

#include <functional>
#include <memory>
#include <queue>
#include <unordered_set>
#include <vector>

#include "autodiff/concepts.h"
#include "autodiff/ops.h"

namespace grad {

template <Numeric T>
class Node : public std::enable_shared_from_this<Node<T>> {
   public:
    using Expression = std::shared_ptr<Node<T>>;
    // P bad hack to publically expose this lol
    using SubexprContainerT = std::unordered_set<Expression>;
   private:
    using BackpropFnType = std::function<void()>;

   public:
    /**************************************
                    Ctors
    ***************************************/
    Node(T value) : value_{value}, op_{Op::CONSTANT}, backprop_fn_{init_backprop_fn()} {}
    Node(T value, Op op) : value_{value}, op_{op}, backprop_fn_{init_backprop_fn()} {}
    Node(T value, Op op, SubexprContainerT inputs)
        : value_{value}, op_{op}, inputs_{std::move(inputs)}, backprop_fn_{init_backprop_fn()} {}

    /**************************************
                   Backprop
    ***************************************/
    void backward() {
        std::vector<Expression> sorted_nodes = input_topological_ordering();

        for (auto& node : sorted_nodes) {
            node->set_grad(0);
        }

        set_grad(1);

        for (auto& subexpr : sorted_nodes) {
            subexpr->backprop_fn_();
        }
    }

    /**************************************
            Getters and setters
    ***************************************/
    T value() const { return value_; }
    void set_value(T value) { value_ = value; }

    T grad() const { return grad_; }
    void accumulate_grad(T grad) { grad_ += grad; }
    void set_grad(T grad) { grad_ = grad; }
    void zero_grad() { grad_ = 0; }

    const SubexprContainerT& get_inputs() const { return inputs_; }

    void set_backprop_fn(BackpropFnType fn) { backprop_fn_ = fn; }

    /**************************************
            Arithmetic operations
    ***************************************/
    Expression operator-() {
        const T operation_result = -1 * value();

        Expression new_expr = std::make_shared<Node<T>>(operation_result, Op::NEGATE);

        // grad is a no-op here.

        return new_expr;
    }

    Expression operator+(const Expression& other) {
        const T operation_result = value() + other->value();

        Expression new_expr = std::make_shared<Node<T>>(
            operation_result, Op::ADD, SubexprContainerT{this->shared_from_this(), other});

        Node<T>* weak_ref = new_expr.get();
        new_expr->backprop_fn_ = [this, other, weak_ref]() {
            this->accumulate_grad(1 * weak_ref->grad());
            other->accumulate_grad(1 * weak_ref->grad());
        };
        return new_expr;
    }

    Expression operator-(const Expression& other) {
        return this->shared_from_this() + other->operator-();
    }

    Expression operator*(const Expression& other) {
        const T operation_result = value() * other->value();

        Expression new_expr = std::make_shared<Node<T>>(
            operation_result, Op::MUL, SubexprContainerT{this->shared_from_this(), other});

        Node<T>* weak_ref = new_expr.get();
        new_expr->backprop_fn_ = [this, other, weak_ref]() {
            this->accumulate_grad(other->value() * weak_ref->grad());
            other->accumulate_grad(this->value() * weak_ref->grad());
        };
        return new_expr;
    }

    Expression operator/(const Expression& other) {
        // Doing the full expression for division is annoying, so use pow
        // to get it done for us
        return this->shared_from_this() * other->pow(std::make_shared<Node<T>>(-1));
    }

    Expression pow(const Expression& other) {
        const T operation_result = std::pow(value(), other->value());

        Expression new_expr = std::make_shared<Node<T>>(
            operation_result, Op::POW, SubexprContainerT{this->shared_from_this(), other});

        Node<T>* weak_ref = new_expr.get();
        // a = x^b
        // da/dx = b * x ^ (b - 1)
        new_expr->backprop_fn_ = [this, other, weak_ref]() {
            this->accumulate_grad(other->value() * std::pow(this->value(), other->value() - 1) *
                                  weak_ref->grad());
        };
        return new_expr;
    }

    /**************************************
         Derived Arithmetic operations
    ***************************************/
    // TODO - add unary negation (doesn't work)
    Expression operator+(T scalar) {
        return this->shared_from_this() + std::make_shared<Node<T>>(scalar);
    }
    Expression operator-(T scalar) {
        return this->shared_from_this() - std::make_shared<Node<T>>(scalar);
    }
    Expression operator*(T scalar) {
        return this->shared_from_this() * std::make_shared<Node<T>>(scalar);
    }
    Expression operator/(T scalar) {
        return this->shared_from_this() / std::make_shared<Node<T>>(scalar);
    }
    Expression pow(T scalar) {
        return this->shared_from_this()->pow(std::make_shared<Node<T>>(scalar));
    }

    friend Expression operator+(const Expression& lhs, T scalar) {
        return (*lhs) + scalar;
    }
    friend Expression operator-(const Expression& lhs, T scalar) {
        return (*lhs) - scalar;
    }
    friend Expression operator*(const Expression& lhs, T scalar) {
        return (*lhs) * scalar;
    }
    friend Expression operator/(const Expression& lhs, T scalar) {
        return (*lhs) / scalar;
    }
    friend Expression pow(const Expression& lhs, T scalar) {
        return lhs.pow(scalar);
    }

    friend Expression operator+(T scalar, const Expression& rhs) {
        return std::make_shared<Node<T>>(scalar) + rhs;
    }
    friend Expression operator-(T scalar, const Expression& rhs) {
        return std::make_shared<Node<T>>(scalar) - rhs;
    }
    friend Expression operator*(T scalar, const Expression& rhs) {
        return std::make_shared<Node<T>>(scalar) * rhs;
    }
    friend Expression operator/(T scalar, const Expression& rhs) {
        return std::make_shared<Node<T>>(scalar) / rhs;
    }
    friend Expression pow(T scalar, const Expression& rhs) {
        Expression pow_base = std::make_shared<Node<T>>(scalar);
        return pow_base.pow(rhs);
    }

    friend Expression operator+(const Expression& lhs, const Expression& rhs) {
        return (*lhs) + rhs;
    }
    friend Expression operator-(const Expression& lhs, const Expression& rhs) {
        return (*lhs) - rhs;
    }
    friend Expression operator*(const Expression& lhs, const Expression& rhs) {
        return (*lhs) * rhs;
    }
    friend Expression operator/(const Expression& lhs, const Expression& rhs) {
        return (*lhs) / rhs;
    }
    friend Expression pow(const Expression& lhs, const Expression& rhs) {
        return lhs.pow(rhs);
    }

   private:
    decltype(auto) init_backprop_fn() {
        return []() {};
    }

    std::vector<Expression> input_topological_ordering() {
        std::vector<Expression> sorted;

        std::unordered_set<Expression> visited;
        std::queue<Expression> q;
        // start from the current expression, working backwards
        auto start = this->shared_from_this();
        q.push(start);
        visited.insert(start);

        while (!q.empty()) {
            auto subexpr = q.front();
            q.pop();
            sorted.push_back(subexpr);
            for (const auto& next_expr : subexpr->get_inputs()) {
                if (visited.find(next_expr) != visited.end()) {
                    continue;
                }
                q.push(next_expr);
                visited.insert(next_expr);
            }
        }

        return sorted;
    }

    T value_{0};
    Op op_{Op::UNKNOWN};

    T grad_{0};
    SubexprContainerT inputs_{};
    BackpropFnType backprop_fn_{};
};

template <Numeric T>
using Expression = std::shared_ptr<Node<T>>;

using ExpressionF = Expression<float>;
using ExpressionD = Expression<double>;

template <Numeric T>
Expression<T> constant(T value) {
    return std::make_shared<Node<T>>(value);
}

}  // namespace grad
