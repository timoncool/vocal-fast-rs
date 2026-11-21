#pragma once

#include <cstddef>
#include <cstring>

namespace vocal::cpu {

inline void upload_projection_rows(void *destination, size_t stride,
                                   const void *source, size_t rows,
                                   size_t columns) {
  const auto row_bytes = columns * sizeof(float);
  auto *padded = static_cast<unsigned char *>(destination);
  const auto *packed = static_cast<const unsigned char *>(source);
  for (size_t row = 0; row < rows; ++row)
    std::memcpy(padded + row * stride, packed + row * row_bytes, row_bytes);
}

inline void download_projection_rows(float *destination, const void *source,
                                     size_t stride, size_t rows, size_t columns,
                                     const float *bias) {
  const auto *padded = static_cast<const unsigned char *>(source);
  for (size_t row = 0; row < rows; ++row) {
    auto *output = destination + row * columns;
    std::memcpy(output, padded + row * stride, columns * sizeof(float));
    if (bias)
      for (size_t column = 0; column < columns; ++column)
        output[column] += bias[column];
  }
}

} // namespace vocal::cpu
