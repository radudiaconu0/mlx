// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/backend/rocm/kernels/utils.h"

namespace mlx::core::mxrocm {

struct Add {
  template <typename T>
  __device__ T operator()(T x, T y) {
    return x + y;
  }
};

struct FloorDivide {
  template <typename T>
  __device__ T operator()(T x, T y) {
    if constexpr (std::is_integral<T>::value) {
      return x / y;
    } else {
      return trunc(x / y);
    }
  }
};

struct Divide {
  template <typename T>
  __device__ T operator()(T x, T y) {
    return x / y;
  }
};

struct Remainder {
  template <typename T>
  __device__ T operator()(T x, T y) {
    if constexpr (std::is_integral<T>::value) {
      if constexpr (std::is_signed<T>::value) {
        auto r = x % y;
        if (r != 0 && (r < 0 != y < 0)) {
          r += y;
        }
        return r;
      } else {
        return x % y;
      }
    } else if constexpr (std::is_same<T, hipFloatComplex>::value) {
      return x % y;
    } else {
      T r = fmod(x, y);
      if (r != 0 && (r < 0 != y < 0)) {
        r = r + y;
      }
      return r;
    }
  }
};

struct Equal {
  template <typename T>
  __device__ bool operator()(T x, T y) {
    return x == y;
  }
};

struct NaNEqual {
  template <typename T>
  __device__ bool operator()(T x, T y) {
    if constexpr (std::is_same<T, hipFloatComplex>::value) {
      return x == y ||
          (isnan(x.x) && isnan(y.x) && isnan(x.y) && isnan(y.y)) ||
          (x.x == y.x && isnan(x.y) && isnan(y.y)) ||
          (isnan(x.x) && isnan(y.x) && x.y == y.y);
    } else {
      return x == y || (isnan(x) && isnan(y));
    }
  }
};

struct Greater {
  template <typename T>
  __device__ bool operator()(T x, T y) {
    return x > y;
  }
};

struct GreaterEqual {
  template <typename T>
  __device__ bool operator()(T x, T y) {
    return x >= y;
  }
};

struct Less {
  template <typename T>
  __device__ bool operator()(T x, T y) {
    return x < y;
  }
};

struct LessEqual {
  template <typename T>
  __device__ bool operator()(T x, T y) {
    return x <= y;
  }
};

struct LogAddExp {
  template <typename T>
  __device__ T operator()(T x, T y) {
    if (isnan(x) || isnan(y)) {
      return std::numeric_limits<T>::quiet_NaN();
    }
    T maxval = max(x, y);
    T minval = min(x, y);
    return (minval == negative_infinite_value<T>() ||
            maxval == infinite_value<T>())
        ? maxval
        : (maxval + log1p(expf(minval - maxval)));
  };
};

struct Maximum {
  template <typename T>
  __device__ T operator()(T x, T y) {
    if constexpr (std::is_integral<T>::value) {
      return max(x, y);
    } else if constexpr (std::is_same<T, hipFloatComplex>::value) {
      if (isnan(x.x) || isnan(x.y)) {
        return x;
      }
      return x > y ? x : y;
    } else {
      if (isnan(x)) {
        return x;
      }
      return x > y ? x : y;
    }
  }
};

struct Minimum {
  template <typename T>
  __device__ T operator()(T x, T y) {
    if constexpr (std::is_integral<T>::value) {
      return min(x, y);
    } else if constexpr (std::is_same<T, hipFloatComplex>::value) {
      if (isnan(x.x) || isnan(x.y)) {
        return x;
      }
      return x < y ? x : y;
    } else {
      if (isnan(x)) {
        return x;
      }
      return x < y ? x : y;
    }
  }
};

struct Multiply {
  template <typename T>
  __device__ T operator()(T x, T y) {
    return x * y;
  }
};

struct NotEqual {
  template <typename T>
  __device__ bool operator()(T x, T y) {
    if constexpr (std::is_same<T, hipFloatComplex>::value) {
      return x.x != y.x || x.y != y.y;
    } else {
      return x != y;
    }
  }
};

struct Power {
  template <typename T>
  __device__ T operator()(T base, T exp) {
    if constexpr (std::is_integral<T>::value) {
      T res = 1;
      while (exp) {
        if (exp & 1) {
          res *= base;
        }
        exp >>= 1;
        base *= base;
      }
      return res;
    } else if constexpr (std::is_same<T, hipFloatComplex>::value) {
      auto x_theta = atan2f(base.y, base.x);
      auto x_ln_r = 0.5f * logf(base.x * base.x + base.y * base.y);
      auto mag = expf(exp.x * x_ln_r - exp.y * x_theta);
      auto phase = exp.y * x_ln_r + exp.x * x_theta;
      return {mag * cosf(phase), mag * sinf(phase)};
    } else {
      return powf(base, exp);
    }
  }
};

struct Subtract {
  template <typename T>
  __device__ T operator()(T x, T y) {
    return x - y;
  }
};

struct LogicalAnd {
  template <typename T>
  __device__ T operator()(T x, T y) {
    return x && y;
  };
};

struct LogicalOr {
  template <typename T>
  __device__ T operator()(T x, T y) {
    return x || y;
  };
};

struct BitwiseAnd {
  template <typename T>
  __device__ T operator()(T x, T y) {
    return x & y;
  };
};

struct BitwiseOr {
  template <typename T>
  __device__ T operator()(T x, T y) {
    return x | y;
  };
};

struct BitwiseXor {
  template <typename T>
  __device__ T operator()(T x, T y) {
    return x ^ y;
  };
};

struct LeftShift {
  template <typename T>
  __device__ T operator()(T x, T y) {
    return x << y;
  };
};

struct RightShift {
  template <typename T>
  __device__ T operator()(T x, T y) {
    return x >> y;
  };
};

struct ArcTan2 {
  template <typename T>
  __device__ T operator()(T y, T x) {
    return atan2f(y, x);
  }
};

struct DivMod {
  template <typename T>
  __device__ std::array<T, 2> operator()(T x, T y) {
    return {FloorDivide{}(x, y), Remainder{}(x, y)};
  };
};

} // namespace mlx::core::mxrocm