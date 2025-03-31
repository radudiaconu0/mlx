// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/backend/rocm/kernels/utils.h"

namespace mlx::core::mxrocm {

struct Abs {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (std::is_unsigned<T>::value) {
      return x;
    } else if constexpr (std::is_same<T, hipFloatComplex>::value) {
      return make_hipFloatComplex(sqrt(x.x * x.x + x.y * x.y), 0);
    } else {
      return abs(x);
    }
  }
};

struct ArcCos {
  template <typename T>
  __device__ T operator()(T x) {
    return acos(x);
  }
};

struct ArcCosh {
  template <typename T>
  __device__ T operator()(T x) {
    return acosh(x);
  }
};

struct ArcSin {
  template <typename T>
  __device__ T operator()(T x) {
    return asin(x);
  }
};

struct ArcSinh {
  template <typename T>
  __device__ T operator()(T x) {
    return asinh(x);
  }
};

struct ArcTan {
  template <typename T>
  __device__ T operator()(T x) {
    return atan(x);
  }
};

struct ArcTanh {
  template <typename T>
  __device__ T operator()(T x) {
    return atanh(x);
  }
};

struct BitwiseInvert {
  template <typename T>
  __device__ T operator()(T x) {
    return ~x;
  }
};

struct Ceil {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (std::is_integral<T>::value) {
      return x;
    } else {
      return ceil(x);
    }
  }
};

struct Conjugate {
  __device__ hipFloatComplex operator()(hipFloatComplex x) {
    return make_hipFloatComplex(x.x, -x.y);
  }
};

struct Cos {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (std::is_same<T, hipFloatComplex>::value) {
      return make_hipFloatComplex(
          cos(x.x) * cosh(x.y),
          -sin(x.x) * sinh(x.y));
    } else {
      return cos(x);
    }
  }
};

struct Cosh {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (std::is_same<T, hipFloatComplex>::value) {
      return make_hipFloatComplex(
          cosh(x.x) * cos(x.y),
          sinh(x.x) * sin(x.y));
    } else {
      return cosh(x);
    }
  }
};

struct Erf {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (std::is_same<T, __half>::value) {
      return __float2half(erf(__half2float(x)));
    } else if constexpr (std::is_same<T, hip_bfloat16>::value) {
      return __float2bfloat16(erf(__bfloat162float(x)));
    } else {
      return erf(x);
    }
  }
};

struct ErfInv {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (std::is_same<T, __half>::value) {
      return __float2half(erfinv(__half2float(x)));
    } else if constexpr (std::is_same<T, hip_bfloat16>::value) {
      return __float2bfloat16(erfinv(__bfloat162float(x)));
    } else {
      return erfinv(x);
    }
  }
};

struct Exp {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (std::is_same<T, hipFloatComplex>::value) {
      auto m = exp(x.x);
      return make_hipFloatComplex(m * cos(x.y), m * sinh(x.y));
    } else {
      return exp(x);
    }
  }
};

struct Expm1 {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (std::is_same<T, __half>::value) {
      return __float2half(expm1(__half2float(x)));
    } else if constexpr (std::is_same<T, hip_bfloat16>::value) {
      return __float2bfloat16(expm1(__bfloat162float(x)));
    } else {
      return expm1(x);
    }
  }
};

struct Floor {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (std::is_integral<T>::value) {
      return x;
    } else {
      return floor(x);
    }
  }
};

struct Imag {
  __device__ float operator()(hipFloatComplex x) {
    return x.y;
  }
};

struct Log {
  template <typename T>
  __device__ T operator()(T x) {
    return log(x);
  }
};

struct Log2 {
  template <typename T>
  __device__ T operator()(T x) {
    return log2(x);
  }
};

struct Log10 {
  template <typename T>
  __device__ T operator()(T x) {
    return log10(x);
  }
};

struct Log1p {
  template <typename T>
  __device__ T operator()(T x) {
    return log1p(x);
  }
};

struct LogicalNot {
  __device__ bool operator()(bool x) {
    return !x;
  }
};

struct Negative {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (std::is_same<T, hipFloatComplex>::value) {
      return make_hipFloatComplex(-x.x, -x.y);
    } else {
      return -x;
    }
  }
};

struct Real {
  __device__ float operator()(hipFloatComplex x) {
    return x.x;
  }
};

struct Round {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (std::is_same<T, hipFloatComplex>::value) {
      return make_hipFloatComplex(rint(x.x), rint(x.y));
    } else {
      return rint(x);
    }
  }
};

struct Rsqrt {
  template <typename T>
  __device__ T operator()(T x) {
    return rsqrt(x);
  }
};

struct Sigmoid {
  template <typename T>
  __device__ T operator()(T x) {
    T y = 1 / (1 + exp(-abs(x)));
    return (x < 0) ? 1 - y : y;
  }
};

struct Sign {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (std::is_unsigned<T>::value) {
      return x != 0;
    } else if constexpr (std::is_same<T, hipFloatComplex>::value) {
      if (x.x == 0 && x.y == 0) {
        return x;
      } else {
        float magnitude = sqrt(x.x * x.x + x.y * x.y);
        return make_hipFloatComplex(x.x / magnitude, x.y / magnitude);
      }
    } else if constexpr (std::is_same<T, hip_bfloat16>::value) {
      return static_cast<float>((x > T(0.f)) - (x < T(0.f)));
    } else {
      return (x > T(0)) - (x < T(0));
    }
  }
};

struct Sin {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (std::is_same<T, hipFloatComplex>::value) {
      return make_hipFloatComplex(
          sin(x.x) * cosh(x.y),
          cos(x.x) * sinh(x.y));
    } else {
      return sin(x);
    }
  }
};

struct Sinh {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (std::is_same<T, hipFloatComplex>::value) {
      return make_hipFloatComplex(
          sinh(x.x) * cos(x.y),
          cosh(x.x) * sin(x.y));
    } else {
      return sinh(x);
    }
  }
};

struct Square {
  template <typename T>
  __device__ T operator()(T x) {
    return x * x;
  }
};

struct Sqrt {
  template <typename T>
  __device__ T operator()(T x) {
    return sqrt(x);
  }
};

struct Tan {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (std::is_same<T, hipFloatComplex>::value) {
      float tan_a = tan(x.x);
      float tanh_b = tanh(x.y);
      float t1 = tan_a * tanh_b;
      float denom = 1.f + t1 * t1;
      return make_hipFloatComplex(
          (tan_a - tanh_b * t1) / denom,
          (tanh_b + tan_a * t1) / denom);
    } else {
      return tan(x);
    }
  }
};

struct Tanh {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (std::is_same<T, hipFloatComplex>::value) {
      float tanh_a = tanh(x.x);
      float tan_b = tan(x.y);
      float t1 = tanh_a * tan_b;
      float denom = 1.f + t1 * t1;
      return make_hipFloatComplex(
          (tanh_a + tan_b * t1) / denom,
          (tan_b - tanh_a * t1) / denom);
    } else {
      return tanh(x);
    }
  }
};

} // namespace mlx::core::mxrocm