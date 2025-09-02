#include <gtest/gtest.h>

#include <numbers>

#include "autodiff/functions.h"

TEST(AutodiffTest, ToStringTest) {
    auto x = grad::constant(1.f) + grad::variable<float>("x");
    auto res = x->to_string();
    EXPECT_EQ(res, "ADD(Const(1.000000), Var(x))");
}

TEST(AutodiffTest, ConstantGradientTestSigmoid) {
    auto x = grad::constant(1.f);

    auto sigmoid = [](const grad::ExpressionF& x) -> grad::ExpressionF {
        return 1.f / (1 + grad::exp(-1 * x));
    }(x);

    sigmoid->get_gradients();
    EXPECT_NEAR(sigmoid->value(), 0.731059f, 0.0001f);
    EXPECT_FLOAT_EQ(x->grad(), 0.196612f);
}


TEST(AutodiffTest, VariableGradientTestTrig) {
    auto x = grad::variable<float>("x");
    auto y = grad::variable<float>("y");

    auto trig_expression = []<Numeric T>(const grad::ExpressionPtr<T>& x, const grad::ExpressionPtr<T>& y) -> grad::ExpressionF  {
        auto mul = x * y;
        auto expr = grad::cos(-1 * mul);
        return expr;
    }(x, y);

    std::unordered_map<std::string, grad::ExpressionF> variables {
        {"x", grad::constant(static_cast<float>(std::numbers::pi))},
        {"y", grad::constant(0.25f)},
    };

    trig_expression->apply_variables(variables);
    ASSERT_NEAR(x->value(), static_cast<float>(std::numbers::pi), 0.0001f);
    ASSERT_NEAR(y->value(), 0.25f, 0.0001f);

    float result = trig_expression->evaluate();
    EXPECT_NEAR(result, 0.707106f, 0.0001f);

    // Have to evaluate before calling get_gradients to populate values.
    // Also can't call get_gradients before applying variables.
    trig_expression->get_gradients();

    ASSERT_NEAR(trig_expression->grad(), 1.f, 0.0001f);
    EXPECT_NEAR(x->grad(), -0.1767766953f, 0.0001f);
    EXPECT_NEAR(y->grad(), -2.22144146908f, 0.0001f);
}
