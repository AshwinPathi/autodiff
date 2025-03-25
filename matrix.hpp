#pragma once

#include "tensor.hpp"

namespace ml {

// Effectively an alias for 2 dimensional matrix
template <typename T> requires std::is_arithmetic_v<T>
class Matrix : public Tensor<T> {
public:
  Matrix(size_t size_dim1, size_t size_dim2) : Tensor<T>{{size_dim1, size_dim2}} {}

  // 1-d buffer construction
  Matrix(size_t size_dim1, size_t size_dim2, const std::vector<T>& buffer) : Tensor<T>{{size_dim1, size_dim2}, buffer} {}
  Matrix(size_t size_dim1, size_t size_dim2, std::vector<T>&& buffer) : Tensor<T>{{size_dim1, size_dim2}, std::move(buffer)} {}

  // 2-d buffer construction from "canonical" 2d vector
  Matrix(const std::vector<std::vector<T>>& buffer) : Tensor<T>{{buffer.size(), buffer[0].size()}, flatten(buffer)} {}

  size_t row_dim() const {
    return this->shape_[0];
  }

  size_t col_dim() const {
    return this->shape_[1];
  }

private:
  typename Tensor<T>::underlying_buffer_type flatten(const std::vector<std::vector<T>>& buffer) {
    typename Matrix<T>::underlying_buffer_type flattened_buffer;
    for(const auto &v: buffer) {
      flattened_buffer.insert(flattened_buffer.end(), v.begin(), v.end());
    }
    return flattened_buffer;
  }
};

} // namespace ml
