#pragma once

#include "tensor.hpp"
#include <vector>

namespace ml {

// Effectively an alias for tensor, but with some 1-d specific utils.
template <typename T> requires std::is_arithmetic_v<T>
class Vector : public Tensor<T> {
public:
  Vector(size_t size) : Tensor<T>{{size}} {}
  Vector(size_t size, const std::vector<T>& buffer) : Tensor<T>{{size}, buffer} {}
  Vector(size_t size, std::vector<T>&& buffer) : Tensor<T>{{size}, std::move(buffer)} {}
};

} // namespace ml
