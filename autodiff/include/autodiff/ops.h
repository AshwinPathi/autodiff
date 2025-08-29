#pragma once

#include "autodiff/concepts.h"

namespace grad {

enum class Op : uint8_t {
    UNKNOWN = 0,
    CONSTANT,
    VARIABLE,
    ADD,
    SUB,
    MUL,
    DIV,
    NEGATE,
    POW,

    SIN,
    COS,
    EXP,
    TAN,
    TANH,
    LN,
};

bool is_unary_op(Op op) {
    switch (op) {
        case Op::NEGATE:
        case Op::SIN:
        case Op::COS:
        case Op::EXP:
        case Op::TAN:
        case Op::TANH:
        case Op::LN:
            return true;
        default:
            return false;
    }
}

bool is_binary_op(Op op) {
    switch (op) {
        case Op::ADD:
        case Op::SUB:
        case Op::MUL:
        case Op::DIV:
        case Op::POW:
            return true;
        default:
            return false;
    }
}

template <Numeric T>
T evaluate_unary_op(Op op, T input) {
    switch (op) {
        case Op::NEGATE:
            return -input;
        case Op::SIN:
            return std::sin(input);
        case Op::COS:
            return std::cos(input);
        case Op::EXP:
            return std::exp(input);
        case Op::TAN:
            return std::tan(input);
        case Op::TANH:
            return std::tanh(input);
        case Op::LN:
            return std::log(input);
        default:
            throw std::runtime_error("Unknown unary operation");
    }
}

template <Numeric T>
T evaluate_binary_op(Op op, T input1, T input2) {
    switch (op) {
        case Op::ADD:
            return input1 + input2;
        case Op::SUB:
            return input1 - input2;
        case Op::MUL:
            return input1 * input2;
        case Op::DIV:
            return input1 / input2;
        case Op::POW:
            return std::pow(input1, input2);
        default:
            throw std::runtime_error("Unknown binary operation");
    }
}

}  // namespace grad
