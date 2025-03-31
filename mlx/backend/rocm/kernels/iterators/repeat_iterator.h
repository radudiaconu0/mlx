// Copyright © 2025 Apple Inc.

#pragma once

#include <rocthrust/iterator/iterator_adaptor.h>

namespace mlx::core::mxrocm {

// Always return the value of initial iterator after advancements.
template <typename Iterator>
class repeat_iterator
    : public rocthrust::iterator_adaptor<
          repeat_iterator<Iterator>,
          Iterator,
          typename rocthrust::iterator_value<Iterator>::type,
          rocthrust::use_default,
          typename rocthrust::iterator_reference<Iterator>::type,
          typename rocthrust::iterator_difference<Iterator>::type> {
public:
  using super_t = rocthrust::iterator_adaptor<
      repeat_iterator<Iterator>,
      Iterator,
      typename rocthrust::iterator_value<Iterator>::type,
      rocthrust::use_default,
      typename rocthrust::iterator_reference<Iterator>::type,
      typename rocthrust::iterator_difference<Iterator>::type>;

  using reference = typename super_t::reference;
  using difference_type = typename super_t::difference_type;

  __host__ __device__ repeat_iterator(Iterator it) : super_t(it), it_(it) {}

private:
  friend class rocthrust::iterator_core_access;

  // The dereference is device-only to avoid accidental running in host.
  __device__ typename super_t::reference dereference() const {
    return *it_;
  }

  Iterator it_;
};

template <typename Iterator>
__host__ __device__ auto make_repeat_iterator(Iterator it) {
  return repeat_iterator<Iterator>(it);
}

} // namespace mlx::core::mxrocm