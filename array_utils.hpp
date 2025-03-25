#pragma once

#include "array.hpp"
#include "utils.hpp"

namespace ml {

template <typename T>
T dot(const Vector<T>& vector1, const Vector<T>& vector2) {
  assertm(vector1.shape()[0] == vector2.shape()[0], "Can only dot product two vectors of the same size.");

  T result = 0;
  for (int i = 0; i < vector1.shape()[0]; i++) {
    result += vector1[i] * vector2[i];
  }

  return result;
}

} // namespace ml
