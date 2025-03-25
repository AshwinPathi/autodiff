#include <iostream>

#include "lazy.hpp"

int main() {

  ml::Vector<float> v1(3, {1.3f, 2.1f, 3.6f});
  ml::Vector<float> v2(3, {2.3f, 1.8f, 0.7f});

  for (const auto& elem : v1) {
    std::cout << elem << "\n";
  }

  std::cout << "dot: " << dot(v1, v2) << "\n";

  return 0;
}