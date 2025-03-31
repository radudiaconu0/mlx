// Copyright © 2025 Apple Inc.
// Copyright © 2017-2024 The Simons Foundation, Inc.
//
// FINUFFT is licensed under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance with the
// License.  You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Adapted for ROCm from
// https://github.com/flatironinstitute/finufft/blob/main/include/cufinufft/contrib/helper_math.h

#pragma once

#include <hip/hip_complex.h>

// This header provides operator overloads for hipComplex types.

__forceinline__ __host__ __device__ hipDoubleComplex
operator+(const hipDoubleComplex& a, const hipDoubleComplex& b) {
  return make_hipDoubleComplex(a.x + b.x, a.y + b.y);
}

__forceinline__ __host__ __device__ hipDoubleComplex
operator-(const hipDoubleComplex& a, const hipDoubleComplex& b) {
  return make_hipDoubleComplex(a.x - b.x, a.y - b.y);
}

__forceinline__ __host__ __device__ hipDoubleComplex
operator*(const hipDoubleComplex& a, const hipDoubleComplex& b) {
  return make_hipDoubleComplex(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

__forceinline__ __host__ __device__ hipDoubleComplex
operator/(const hipDoubleComplex& a, const hipDoubleComplex& b) {
  double denom = b.x * b.x + b.y * b.y;
  return make_hipDoubleComplex(
      (a.x * b.x + a.y * b.y) / denom,
      (a.y * b.x - a.x * b.y) / denom);
}

__forceinline__ __host__ __device__ hipDoubleComplex
operator%(const hipDoubleComplex& a, const hipDoubleComplex& b) {
  double r = a.x - (floor(a.x / b.x) * b.x);
  double i = a.y - (floor(a.y / b.y) * b.y);
  return make_hipDoubleComplex(r, i);
}

__forceinline__ __host__ __device__ bool operator==(
    const hipDoubleComplex& a,
    const hipDoubleComplex& b) {
  return a.x == b.x && a.y == b.y;
}

__forceinline__ __host__ __device__ bool operator!=(
    const hipDoubleComplex& a,
    const hipDoubleComplex& b) {
  return !(a == b);
}

__forceinline__ __host__ __device__ bool operator>(
    const hipDoubleComplex& a,
    const hipDoubleComplex& b) {
  double mag_a = sqrt(a.x * a.x + a.y * a.y);
  double mag_b = sqrt(b.x * b.x + b.y * b.y);
  return mag_a > mag_b;
}

__forceinline__ __host__ __device__ bool operator>=(
    const hipDoubleComplex& a,
    const hipDoubleComplex& b) {
  return a > b || a == b;
}

__forceinline__ __host__ __device__ bool operator<(
    const hipDoubleComplex& a,
    const hipDoubleComplex& b) {
  return b > a;
}

__forceinline__ __host__ __device__ bool operator<=(
    const hipDoubleComplex& a,
    const hipDoubleComplex& b) {
  return b > a || a == b;
}

__forceinline__ __host__ __device__ hipDoubleComplex
operator+(const hipDoubleComplex& a, double b) {
  return make_hipDoubleComplex(a.x + b, a.y);
}

__forceinline__ __host__ __device__ hipDoubleComplex
operator+(double a, const hipDoubleComplex& b) {
  return make_hipDoubleComplex(a + b.x, b.y);
}

__forceinline__ __host__ __device__ hipDoubleComplex
operator-(const hipDoubleComplex& a, double b) {
  return make_hipDoubleComplex(a.x - b, a.y);
}

__forceinline__ __host__ __device__ hipDoubleComplex
operator-(double a, const hipDoubleComplex& b) {
  return make_hipDoubleComplex(a - b.x, -b.y);
}

__forceinline__ __host__ __device__ hipDoubleComplex
operator*(const hipDoubleComplex& a, double b) {
  return make_hipDoubleComplex(a.x * b, a.y * b);
}

__forceinline__ __host__ __device__ hipDoubleComplex
operator*(double a, const hipDoubleComplex& b) {
  return make_hipDoubleComplex(a * b.x, a * b.y);
}

__forceinline__ __host__ __device__ hipDoubleComplex
operator/(const hipDoubleComplex& a, double b) {
  return make_hipDoubleComplex(a.x / b, a.y / b);
}

__forceinline__ __host__ __device__ hipDoubleComplex
operator/(double a, const hipDoubleComplex& b) {
  double denom = b.x * b.x + b.y * b.y;
  return make_hipDoubleComplex(
      (a * b.x) / denom, (-a * b.y) / denom);
}

// Similar operators for hipFloatComplex
__forceinline__ __host__ __device__ hipFloatComplex
operator+(const hipFloatComplex& a, const hipFloatComplex& b) {
  return make_hipFloatComplex(a.x + b.x, a.y + b.y);
}

__forceinline__ __host__ __device__ hipFloatComplex
operator-(const hipFloatComplex& a, const hipFloatComplex& b) {
  return make_hipFloatComplex(a.x - b.x, a.y - b.y);
}

__forceinline__ __host__ __device__ hipFloatComplex
operator*(const hipFloatComplex& a, const hipFloatComplex& b) {
  return make_hipFloatComplex(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

__forceinline__ __host__ __device__ hipFloatComplex
operator/(const hipFloatComplex& a, const hipFloatComplex& b) {
  float denom = b.x * b.x + b.y * b.y;
  return make_hipFloatComplex(
      (a.x * b.x + a.y * b.y) / denom,
      (a.y * b.x - a.x * b.y) / denom);
}

__forceinline__ __host__ __device__ hipFloatComplex
operator%(const hipFloatComplex& a, const hipFloatComplex& b) {
  float r = a.x - (floorf(a.x / b.x) * b.x);
  float i = a.y - (floorf(a.y / b.y) * b.y);
  return make_hipFloatComplex(r, i);
}

__forceinline__ __host__ __device__ bool operator==(
    const hipFloatComplex& a,
    const hipFloatComplex& b) {
  return a.x == b.x && a.y == b.y;
}

__forceinline__ __host__ __device__ bool operator!=(
    const hipFloatComplex& a,
    const hipFloatComplex& b) {
  return !(a == b);
}

__forceinline__ __host__ __device__ bool operator>(
    const hipFloatComplex& a,
    const hipFloatComplex& b) {
  float mag_a = sqrt(a.x * a.x + a.y * a.y);
  float mag_b = sqrt(b.x * b.x + b.y * b.y);
  return mag_a > mag_b;
}

__forceinline__ __host__ __device__ bool operator>=(
    const hipFloatComplex& a,
    const hipFloatComplex& b) {
  return a > b || a == b;
}

__forceinline__ __host__ __device__ bool operator<(
    const hipFloatComplex& a,
    const hipFloatComplex& b) {
  return b > a;
}

__forceinline__ __host__ __device__ bool operator<=(
    const hipFloatComplex& a,
    const hipFloatComplex& b) {
  return b > a || a == b;
}

__forceinline__ __host__ __device__ hipFloatComplex
operator+(const hipFloatComplex& a, float b) {
  return make_hipFloatComplex(a.x + b, a.y);
}

__forceinline__ __host__ __device__ hipFloatComplex
operator+(float a, const hipFloatComplex& b) {
  return make_hipFloatComplex(a + b.x, b.y);
}

__forceinline__ __host__ __device__ hipFloatComplex
operator-(const hipFloatComplex& a, float b) {
  return make_hipFloatComplex(a.x - b, a.y);
}

__forceinline__ __host__ __device__ hipFloatComplex
operator-(float a, const hipFloatComplex& b) {
  return make_hipFloatComplex(a - b.x, -b.y);
}

__forceinline__ __host__ __device__ hipFloatComplex
operator*(const hipFloatComplex& a, float b) {
  return make_hipFloatComplex(a.x * b, a.y * b);
}

__forceinline__ __host__ __device__ hipFloatComplex
operator*(float a, const hipFloatComplex& b) {
  return make_hipFloatComplex(a * b.x, a * b.y);
}

__forceinline__ __host__ __device__ hipFloatComplex
operator/(const hipFloatComplex& a, float b) {
  return make_hipFloatComplex(a.x / b, a.y / b);
}

__forceinline__ __host__ __device__ hipFloatComplex
operator/(float a, const hipFloatComplex& b) {
  float denom = b.x * b.x + b.y * b.y;
  return make_hipFloatComplex(
      (a * b.x) / denom, (-a * b.y) / denom);
}