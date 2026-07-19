#pragma once

#include <cfloat>

// Define the integer constants first
constexpr uint32_t FP32_MAX_BITS = 0x7F7FFFFFu;
constexpr uint32_t FP32_NEG_MAX_BITS = 0xFF7FFFFFu;
constexpr uint32_t FP32_INFINITY_BITS = 0x7F800000u;

// TF32 Precision (10-bit mantissa in 32-bit float)
constexpr uint32_t TF32_MAX_BITS = 0x7F7FE000u;
constexpr uint32_t TF32_NEG_MAX_BITS = 0xFF7FE000u;
constexpr uint32_t TF32_INFINITY_BITS = 0x7F800000u;
constexpr uint32_t TF32_NEG_MAX_DIV_2_BITS = 0xFEFFE000u;
// bitmask for extracting TF32 from FP32
constexpr uint32_t TF32_VALID_BITS_MASK = 0xFFFFE000u;



#if __cplusplus >= 202002L

#include <bit>

constexpr float FP32_MAX = std::bit_cast<float>(FP32_MAX_BITS);
constexpr float FP32_NEG_MAX = std::bit_cast<float>(FP32_NEG_MAX_BITS);
constexpr float FP32_INFINITY = std::bit_cast<float>(FP32_INFINITY_BITS);

// TF32 Precision (10-bit mantissa in 32-bit float)
constexpr float TF32_MAX = std::bit_cast<float>(TF32_MAX_BITS);
constexpr float TF32_NEG_MAX = std::bit_cast<float>(TF32_NEG_MAX_BITS);
constexpr float TF32_INFINITY = std::bit_cast<float>(TF32_INFINITY_BITS);
constexpr float TF32_NEG_MAX_DIV_2 = std::bit_cast<float>(TF32_NEG_MAX_DIV_2_BITS);

#else

template <typename T, uint32_t bits>
constexpr T bit_cast_compatible() {
  constexpr uint32_t bits_val = bits;
  return reinterpret_cast<T const&>(bits_val);
}

#define FP32_MAX bit_cast_compatible<float, FP32_MAX_BITS>()
#define FP32_NEG_MAX bit_cast_compatible<float, FP32_NEG_MAX_BITS>()
#define FP32_INFINITY bit_cast_compatible<float, FP32_INFINITY_BITS>()

#define TF32_MAX bit_cast_compatible<float, TF32_MAX_BITS>()
#define TF32_NEG_MAX bit_cast_compatible<float, TF32_NEG_MAX_BITS>()
#define TF32_INFINITY bit_cast_compatible<float, TF32_INFINITY_BITS>()
#define TF32_NEG_MAX_DIV_2 bit_cast_compatible<float, TF32_NEG_MAX_DIV_2_BITS>()

#endif



#if __cplusplus >= 202002L

constexpr float MASK_VALUE = TF32_NEG_MAX_DIV_2;

#else

#define MASK_VALUE (TF32_NEG_MAX_DIV_2)

#endif