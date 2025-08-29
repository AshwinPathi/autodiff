#pragma once

#include <concepts>

template <typename T>
concept Numeric = std::is_arithmetic_v<T>;
