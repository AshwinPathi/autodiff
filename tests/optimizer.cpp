#include <gtest/gtest.h>

#include <numbers>

#include "autodiff/functions.h"
#include "autodiff/optimizer/optimizer.h"
#include "autodiff/optimizer/passes/common_subexpression_elim.h"

TEST(OptimizerTest, TestConstantFolding) {
    using namespace grad;
    using namespace grad::optimizer;

    ExpressionF x = variable<float>("x");

    ExpressionF expr = (x * constant(2.0f)) + (constant(3.0f) * constant(4.0f));

    std::vector<std::shared_ptr<Pass<float>>> passes = {
        std::make_shared<CommonSubexpressionElimPass<float>>()
    };
    ExpressionF optimized_expr = optimize(expr, passes);

    std::unordered_map<std::string, grad::ExpressionF> variables {
        {"x", grad::constant(4.f)},
    };

    expr->apply_variables(variables);
    optimized_expr->apply_variables(variables);

    float original_value = expr->evaluate();
    float optimized_value = optimized_expr->evaluate();

    EXPECT_FLOAT_EQ(original_value, optimized_value);
}
