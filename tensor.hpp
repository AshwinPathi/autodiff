#pragma once

#include "utils.hpp"

#include <vector>
#include <type_traits>
#include <numeric>

namespace ml {

// Very basic (and naive) implementation of a tensor.
template <typename T> requires std::is_arithmetic_v<T>
class Tensor {
public:
  using data_type = T;
  using index_type = std::vector<size_t>;
  using underlying_buffer_type = std::vector<T>;

  //////// Constructors
  Tensor(const std::vector<size_t>& shape, bool zeros) : shape_{shape} {
    // Underlying size of the buffer is the product of all the dimensions, since
    // the underlying buffer is really just a 1-d array thats strided differently
    // based on the shape.
    const size_t buffer_size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());

    buffer_.resize(buffer_size);

    if (zeros) {
      for (int i = 0; i < buffer_.size(); i++) {
        buffer_[0] = static_cast<T>(0);
      }
    }
  }

  Tensor(const std::vector<size_t>& shape) : Tensor(shape, /*zeros=*/false) {}

  Tensor(const std::vector<size_t>& shape, const std::vector<T>& buffer) : shape_{shape}, buffer_{buffer} {}

  Tensor(const std::vector<size_t>& shape, std::vector<T>&& buffer) : shape_{shape}, buffer_{std::move(buffer)} {}

  //////// Copy/Move constructors on other Tensors
  Tensor(const Tensor<T>& other) : shape_{other.shape_}, buffer_{other.buffer_} {}

  Tensor(Tensor<T>&& other) : shape_{std::move(other.shape_)}, buffer_{std::move(other.buffer_)} {}

  Tensor& operator=(const Tensor<T>& other) {
    shape_ = other.shape_;
    buffer_ = other.buffer_;
    return *this;
  }

  Tensor& operator=(Tensor<T>&& other) {
    shape_ = std::move(other.shape_);
    buffer_ = std::move(other.buffer_);
    return *this;
  }

  //////// Destructor
  // Doesn't really need to do anything special, underlying vectors should automatically
  // destruct.
  ~Tensor() = default;

  //////// Operator overloads.
  /// WARNING - these operators return a copy of the results of the operations.
  Tensor<T> operator+(const Tensor<T>& other) {
    Tensor<T> result = *this;
    result += other;
    return result;
  }

  Tensor<T> operator-(const Tensor<T>& other) {
    Tensor<T> result = *this;
    result -= other;
    return result;
  }

  Tensor<T> operator*(const Tensor<T>& other) {
    assertm(shape_ == other.shape(), "Shapes must match for element-wise multiplication");

    Tensor<T> result(shape_);
    for (int i = 0; i < buffer_.size(); i++) {
      result.buffer_[i] = buffer_[i] * other.buffer_[i];
    }

    return result;
  }

  Tensor<T> operator/(const Tensor<T>& other) {
    assertm(shape_ == other.shape(), "Shapes must match for element-wise division");

    Tensor<T> result(shape_);
    for (int i = 0; i < buffer_.size(); i++) {
      result.buffer_[i] = buffer_[i] / other.buffer_[i];
    }

    return result;
  }

  /// These modify the current tensor in place.
  Tensor<T>& operator+=(const Tensor<T>& other) {
    assertm(shape_ == other.shape(), "Shapes must match for addition");

    for (int i = 0; i < buffer_.size(); i++) {
      buffer_[i] += other.buffer_[i];
    }

    return *this;
  }

  Tensor<T>& operator-=(const Tensor<T>& other) {
    assertm(shape_ == other.shape(), "Shapes must match for subtraction");

    for (int i = 0; i < buffer_.size(); i++) {
      buffer_[i] -= other.buffer_[i];
    }

    return *this;
  }

  Tensor<T> operator-() {
    Tensor<T> result(shape_);
    for (int i = 0; i < buffer_.size(); i++) {
      result.buffer_[i] = -buffer_[i];
    }

    return result;
  }

  // Indexing into the tensor
  const data_type& operator[](const index_type& indices) const {
    return buffer_[linearize_index(indices)];
  }

  const data_type& operator[](size_t index) const {
    return buffer_[index];
  }

  data_type& operator[](const index_type& indices) {
    return buffer_[linearize_index(indices)];
  }

  data_type& operator[](size_t index) {
    return buffer_[index];
  }

  //////// Accessors for the underlying data
  std::vector<T>& buffer() {
    return buffer_;
  }

  const std::vector<T>& buffer() const {
    return buffer_;
  }

  const std::vector<size_t>& shape() const {
    return shape_;
  }

  //////// Iterators over the underlying buffer
  auto begin() { return buffer_.begin(); }
  auto end() { return buffer_.end(); }
  auto cbegin() const { return buffer_.begin(); }
  auto cend() const { return buffer_.end(); }
  auto begin() const { return buffer_.begin(); }
  auto end() const { return buffer_.end(); }


  //////// Misc helpers
  size_t size() const {
    return buffer_.size();
  }

protected:
  size_t linearize_index(const std::vector<size_t>& indices) const {
    size_t linear_index = 0;
    size_t stride = 1;

    for (int i = shape_.size() - 1; i >= 0; --i) {
        linear_index += indices[i] * stride;
        stride *= shape_[i];
    }

    return linear_index;
  }

  std::vector<size_t> shape_;
  std::vector<T> buffer_;
};

} // namespace ml
