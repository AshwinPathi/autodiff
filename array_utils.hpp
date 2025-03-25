#pragma once

#include "array.hpp"

namespace ml {

template <typename T>
Tensor<T> dot(const Vector<T>& vector1, const Vector<T>& vector2) {
  T result = 0;
  for (int i = 0; i < result.shape().size(); i++) {
    result += vector1[i] * vector2[i];
  }

  return result;
}

} // namespace ml
