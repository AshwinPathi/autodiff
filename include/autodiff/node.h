#pragma once

#include <functional>
#include <memory>
#include <queue>
#include <unordered_set>
#include <vector>
#include <cmath>

#include "autodiff/concepts.h"
#include "autodiff/ops.h"

namespace grad {

template <Numeric T>
class Node : public std::enable_shared_from_this<Node<T>> {
   public:
    using ExpressionPtr = std::shared_ptr<Node<T>>;
    // P bad hack to publically expose this lol
    using SubexprContainerT = std::vector<ExpressionPtr>;
   private:
    using BackpropFnType = std::function<void()>;

   public:
    /**************************************
                    Ctors
    ***************************************/
    Node(std::string var_name) : op_{Op::VARIABLE}, var_name_{std::move(var_name)}, backprop_fn_{init_backprop_fn()} {}

    Node(T value) : value_{value}, op_{Op::CONSTANT}, backprop_fn_{init_backprop_fn()} {}
    Node(T value, Op op) : value_{value}, op_{op}, backprop_fn_{init_backprop_fn()} {}
    Node(T value, Op op, SubexprContainerT inputs)
        : value_{value}, op_{op}, inputs_{std::move(inputs)}, backprop_fn_{init_backprop_fn()} {}

    /**************************************
                   Backprop
    ***************************************/
    void get_gradients() {
        std::vector<ExpressionPtr> sorted_nodes = input_topological_ordering();

        for (auto& node : sorted_nodes) {
            node->set_grad(0);
        }

        set_grad(1);

        for (auto& subexpr : sorted_nodes) {
            if (subexpr->op_ == Op::VARIABLE) {
                throw std::runtime_error("Cannot backprop on variable "+ subexpr->var_name_ + " without applying a value to it.");
                continue;
            }
            subexpr->backprop_fn_();
        }
    }

    /**************************************
                   Forward
    ***************************************/
    T evaluate() {
        return evaluate_helper(this->shared_from_this());
    }

    void apply_variables(const std::unordered_map<std::string, ExpressionPtr>& values) {
        // TODO - probably not the most efficient way to do this(?)
        if (op_ == Op::VARIABLE) {
            auto it = values.find(var_name_);
            if (it == values.end()) {
                throw std::runtime_error("Variable " + var_name_ + " not found in provided values");
            }
            op_ = Op::CONSTANT;
            value_ = it->second->value();
        }

        for (const auto& subexpr : inputs_) {
            subexpr->apply_variables(values);
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

    void mark_as_const() { op_ = Op::CONSTANT; }
    Op get_op() const { return op_; }

    void clear_inputs() { inputs_.clear(); }

    /**************************************
            Arithmetic operations
    ***************************************/
    ExpressionPtr operator-() {
        const T operation_result = -1 * value();

        ExpressionPtr new_expr = std::make_shared<Node<T>>(operation_result, Op::NEGATE);

        return new_expr;
    }

    ExpressionPtr operator+(const ExpressionPtr& other) {
        const T operation_result = value() + other->value();

        ExpressionPtr new_expr = std::make_shared<Node<T>>(
            operation_result, Op::ADD, SubexprContainerT{this->shared_from_this(), other});

        Node<T>* weak_ref = new_expr.get();
        new_expr->backprop_fn_ = [this, other, weak_ref]() {
            this->accumulate_grad(1 * weak_ref->grad());
            other->accumulate_grad(1 * weak_ref->grad());
        };
        return new_expr;
    }

    ExpressionPtr operator-(const ExpressionPtr& other) {
        return this->shared_from_this() + other->operator-();
    }

    ExpressionPtr operator*(const ExpressionPtr& other) {
        const T operation_result = value() * other->value();

        ExpressionPtr new_expr = std::make_shared<Node<T>>(
            operation_result, Op::MUL, SubexprContainerT{this->shared_from_this(), other});

        Node<T>* weak_ref = new_expr.get();
        new_expr->backprop_fn_ = [this, other, weak_ref]() {
            this->accumulate_grad(other->value() * weak_ref->grad());
            other->accumulate_grad(this->value() * weak_ref->grad());
        };
        return new_expr;
    }

    ExpressionPtr operator/(const ExpressionPtr& other) {
        // Doing the full expression for division is annoying, so use pow
        // to get it done for us
        return this->shared_from_this() * other->pow(std::make_shared<Node<T>>(-1));
    }

    ExpressionPtr pow(const ExpressionPtr& other) {
        const T operation_result = std::pow(value(), other->value());

        ExpressionPtr new_expr = std::make_shared<Node<T>>(
            operation_result, Op::POW, SubexprContainerT{this->shared_from_this(), other});

        Node<T>* weak_ref = new_expr.get();
        // a = x^b
        // da/dx = b * x ^ (b - 1)
        // a = b^x
        // da/dx = log(b) * b^x
        new_expr->backprop_fn_ = [this, other, weak_ref]() {
            this->accumulate_grad(other->value() * std::pow(this->value(), other->value() - 1) *
                                  weak_ref->grad());
            other->accumulate_grad(std::log(this->value()) * std::pow(this->value(), other->value()) * weak_ref->grad());
        };
        return new_expr;
    }

    /**************************************
         Derived Arithmetic operations
    ***************************************/
    // TODO - add unary negation (doesn't work)

    ExpressionPtr operator+(T scalar) {
        return this->shared_from_this() + std::make_shared<Node<T>>(scalar);
    }
    ExpressionPtr operator-(T scalar) {
        return this->shared_from_this() - std::make_shared<Node<T>>(scalar);
    }
    ExpressionPtr operator*(T scalar) {
        return this->shared_from_this() * std::make_shared<Node<T>>(scalar);
    }
    ExpressionPtr operator/(T scalar) {
        return this->shared_from_this() / std::make_shared<Node<T>>(scalar);
    }
    ExpressionPtr pow(T scalar) {
        return this->shared_from_this()->pow(std::make_shared<Node<T>>(scalar));
    }

    friend ExpressionPtr operator+(const ExpressionPtr& lhs, T scalar) {
        return (*lhs) + scalar;
    }
    friend ExpressionPtr operator-(const ExpressionPtr& lhs, T scalar) {
        return (*lhs) - scalar;
    }
    friend ExpressionPtr operator*(const ExpressionPtr& lhs, T scalar) {
        return (*lhs) * scalar;
    }
    friend ExpressionPtr operator/(const ExpressionPtr& lhs, T scalar) {
        return (*lhs) / scalar;
    }
    friend ExpressionPtr pow(const ExpressionPtr& lhs, T scalar) {
        return lhs.pow(scalar);
    }

    friend ExpressionPtr operator+(T scalar, const ExpressionPtr& rhs) {
        return std::make_shared<Node<T>>(scalar) + rhs;
    }
    friend ExpressionPtr operator-(T scalar, const ExpressionPtr& rhs) {
        return std::make_shared<Node<T>>(scalar) - rhs;
    }
    friend ExpressionPtr operator*(T scalar, const ExpressionPtr& rhs) {
        return std::make_shared<Node<T>>(scalar) * rhs;
    }
    friend ExpressionPtr operator/(T scalar, const ExpressionPtr& rhs) {
        return std::make_shared<Node<T>>(scalar) / rhs;
    }

    friend ExpressionPtr operator+(const ExpressionPtr& lhs, const ExpressionPtr& rhs) {
        return (*lhs) + rhs;
    }
    friend ExpressionPtr operator-(const ExpressionPtr& lhs, const ExpressionPtr& rhs) {
        return (*lhs) - rhs;
    }
    friend ExpressionPtr operator*(const ExpressionPtr& lhs, const ExpressionPtr& rhs) {
        return (*lhs) * rhs;
    }
    friend ExpressionPtr operator/(const ExpressionPtr& lhs, const ExpressionPtr& rhs) {
        return (*lhs) / rhs;
    }


    /**************************************
                    Helpers
    ***************************************/
    std::string to_string() const {
        std::string repr = "";
        if (op_ == Op::VARIABLE) {
            repr += "Var(" + var_name_ + ")";
        } else {
            repr += "Op(" + op_to_string(op_) + ", " + std::to_string(value_) + ")";
        }
        return repr;
    }

   private:
    decltype(auto) init_backprop_fn() const {
        return []() {};
    }

    std::vector<ExpressionPtr> input_topological_ordering() {
        std::vector<ExpressionPtr> sorted;

        std::unordered_set<ExpressionPtr> visited;
        std::queue<ExpressionPtr> q;
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

    T evaluate_helper(const ExpressionPtr& expr) {
        if (expr->op_ == Op::CONSTANT) {
            return expr->value_;
        }

        T result{0};

        if (is_unary_op(expr->op_)) {
            result = evaluate_unary_op(expr->op_, evaluate_helper(expr->inputs_[0]));
        } else if (is_binary_op(expr->op_)) {
            result = evaluate_binary_op(expr->op_, evaluate_helper(expr->inputs_[0]), evaluate_helper(expr->inputs_[1]));
        } else {
            // Wish I had reflection here...
            throw std::runtime_error("Cannot evaluate node with op type " + std::to_string(static_cast<int>(expr->op_)));
        }

        expr->set_value(result);
        return result;
    }

    T value_{0};
    Op op_{Op::UNKNOWN};
    std::string var_name_{};

    T grad_{0};
    SubexprContainerT inputs_{};
    BackpropFnType backprop_fn_{};
};

template <Numeric T>
using ExpressionPtr = std::shared_ptr<Node<T>>;

using ExpressionF = ExpressionPtr<float>;
using ExpressionD = ExpressionPtr<double>;

template <Numeric T>
ExpressionPtr<T> constant(T value) {
    return std::make_shared<Node<T>>(value);
}

template <Numeric T>
ExpressionPtr<T> variable(std::string var_name) {
    return std::make_shared<Node<T>>(std::move(var_name));
}

}  // namespace grad
