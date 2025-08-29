#pragma once

#include <cmath>

#include "autodiff/node.h"

namespace grad {

template <Numeric T>
ExpressionPtr<T> pow(T scalar, const ExpressionPtr<T>& rhs) {
    ExpressionPtr<T> pow_base = std::make_shared<Node<T>>(scalar);
    return pow_base->pow(rhs);
}

template <Numeric T>
ExpressionPtr<T> pow(const ExpressionPtr<T>& lhs, const ExpressionPtr<T>& rhs) {
    return lhs.pow(rhs);
}

template <Numeric T>
ExpressionPtr<T> sin(const ExpressionPtr<T>& expr) {
    const T operation_result = std::sin(expr->value());
    ExpressionPtr<T> new_expr = std::make_shared<Node<T>>(operation_result, Op::SIN,
                                                       typename Node<T>::SubexprContainerT{expr});
    Node<T>* weak_ref = new_expr.get();
    new_expr->set_backprop_fn(
        // d/dx sin(x) = cos(x)
        [expr, weak_ref]() { expr->accumulate_grad(std::cos(expr->value()) * weak_ref->grad()); });
    return new_expr;
}

template <Numeric T>
ExpressionPtr<T> cos(const ExpressionPtr<T>& expr) {
    const T operation_result = std::cos(expr->value());
    ExpressionPtr<T> new_expr = std::make_shared<Node<T>>(operation_result, Op::COS,
                                                       typename Node<T>::SubexprContainerT{expr});
    Node<T>* weak_ref = new_expr.get();
    new_expr->set_backprop_fn(
        // d/dx cos(x) = -sin(x)
        [expr, weak_ref]() {  expr->accumulate_grad(-std::sin(expr->value()) * weak_ref->grad()); });
    return new_expr;
}

template <Numeric T>
ExpressionPtr<T> exp(const ExpressionPtr<T>& expr) {
    const T operation_result = std::exp(expr->value());
    ExpressionPtr<T> new_expr = std::make_shared<Node<T>>(operation_result, Op::EXP,
                                                       typename Node<T>::SubexprContainerT{expr});
    Node<T>* weak_ref = new_expr.get();
    new_expr->set_backprop_fn(
        // d/dx exp(x) = exp(x)
        [expr, weak_ref]() { expr->accumulate_grad(std::exp(expr->value()) * weak_ref->grad()); });
    return new_expr;
}

template <Numeric T>
ExpressionPtr<T> tanh(const ExpressionPtr<T>& expr) {
    // Could also make this as a pure expression of exp(x) for fun
    const T operation_result = std::tanh(expr->value());
    ExpressionPtr<T> new_expr = std::make_shared<Node<T>>(operation_result, Op::TANH,
                                                       typename Node<T>::SubexprContainerT{expr});
    
    Node<T>* weak_ref = new_expr.get();
    new_expr->set_backprop_fn(
        // d/dx tanh(x) = 1 - tanh^2(x)
        [expr, weak_ref]() { expr->accumulate_grad((1 - std::pow(std::tanh(expr->value()), 2)) * weak_ref->grad()); });
    return new_expr;
}

template <Numeric T>
ExpressionPtr<T> ln(const ExpressionPtr<T>& expr) {
    const T operation_result = std::log(expr->value());
    ExpressionPtr<T> new_expr = std::make_shared<Node<T>>(operation_result, Op::LN,
                                                       typename Node<T>::SubexprContainerT{expr});
    
    Node<T>* weak_ref = new_expr.get();
    new_expr->set_backprop_fn(
        // d/dx ln(x) = 1 / x
        [expr, weak_ref]() { expr->accumulate_grad(std::pow(expr->value(), -1) * weak_ref->grad()); });
    return new_expr;
}

}  // namespace grad
