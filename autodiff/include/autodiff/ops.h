#pragma once

namespace grad {

enum class Op {
    UNKNOWN = 0,
    CONSTANT,
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
};

}  // namespace grad
