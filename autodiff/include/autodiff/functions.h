#pragma once

#include <cmath>

#include "autodiff/node.h"

namespace grad {

template <Numeric T>
Expression<T> sin(const Expression<T>& expr) {
    const T operation_result = std::sin(expr->value());
    Expression<T> new_expr = std::make_shared<Node<T>>(operation_result, Op::SIN,
                                                       typename Node<T>::SubexprContainerT{expr});
    Node<T>* weak_ref = new_expr.get();
    new_expr->set_backprop_fn(
        [expr, weak_ref]() { expr->accumulate_grad(std::cos(expr->value()) * weak_ref->grad()); });
    return new_expr;
}

template <Numeric T>
Expression<T> cos(const Expression<T>& expr) {
    const T operation_result = std::cos(expr->value());
    Expression<T> new_expr = std::make_shared<Node<T>>(operation_result, Op::COS,
                                                       typename Node<T>::SubexprContainerT{expr});
    Node<T>* weak_ref = new_expr.get();
    new_expr->set_backprop_fn(
        [expr, weak_ref]() { expr->accumulate_grad(-std::sin(expr->value()) * weak_ref->grad()); });
    return new_expr;
}

template <Numeric T>
Expression<T> exp(const Expression<T>& expr) {
    const T operation_result = std::exp(expr->value());
    Expression<T> new_expr = std::make_shared<Node<T>>(operation_result, Op::EXP,
                                                       typename Node<T>::SubexprContainerT{expr});
    Node<T>* weak_ref = new_expr.get();
    new_expr->set_backprop_fn(
        [expr, weak_ref]() { expr->accumulate_grad(std::exp(expr->value()) * weak_ref->grad()); });
    return new_expr;
}

template <Numeric T>
Expression<T> tan(const Expression<T>& expr) {
    const T operation_result = std::tan(expr->value());
    Expression<T> new_expr = std::make_shared<Node<T>>(operation_result, Op::TAN,
                                                       typename Node<T>::SubexprContainerT{expr});
    
    Node<T>* weak_ref = new_expr.get();
    new_expr->set_backprop_fn(
        [expr, weak_ref]() { expr->accumulate_grad(std::pow(std::pow(std::cos(expr->value()), 2), -1) * weak_ref->grad()); });
    return new_expr;
}

template <Numeric T>
Expression<T> tanh(const Expression<T>& expr) {
    const T operation_result = std::tanh(expr->value());
    Expression<T> new_expr = std::make_shared<Node<T>>(operation_result, Op::TANH,
                                                       typename Node<T>::SubexprContainerT{expr});
    
    Node<T>* weak_ref = new_expr.get();
    new_expr->set_backprop_fn(
        [expr, weak_ref]() { expr->accumulate_grad((1 - std::pow(std::tanh(expr->value), 2)) * weak_ref->grad()); });
    return new_expr;
}

}  // namespace grad
