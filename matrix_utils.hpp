#pragma once

#include "matrix.hpp"

namespace ml {

template <typename T>
Tensor<T> matmul(const Matrix<T>& mat1, const Matrix<T>& mat2) {
  Tensor<T> result(mat1.shape(), /*zeros=*/true);

  for (size_t i = 0; i < mat1.shape()[0]; i++) {
    for (size_t j = 0; j < mat2.shape()[1]; j++) {
      for (size_t k = 0; k < mat1.shape()[1]; k++) {
        result[{i, j}] += mat1[{i, k}] * mat2[{k, j}];
      }
    }
  }

  return result;
}

template <typename T>
T det(const Matrix<T>& mat) {
  assertm(mat.row_dim() == mat.col_dim(), "Determinant is only defined for square matrices");

  T result = 0;

  using index_type = typename Tensor<T>::index_type;
  const auto two_by_two_det = [&](const index_type& top_left, const index_type& bottom_right,
                                 const index_type& top_right, const index_type& bottom_left) {
    return mat[top_left] * mat[bottom_right] - mat[top_right] * mat[bottom_left];
  };

  if (mat.row_dim() == 2 && mat.col_dim() == 2) {
    return two_by_two_det({0,0}, {1,1}, {0,1}, {1,0});
  }

  // TODO - generalize for all cases of N x N square matrices.

  return result;
}

} // namespace ml
