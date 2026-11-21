#include "../native/projection_rows.h"

#include <array>
#include <cassert>

int main() {
  using namespace vocal::cpu;
  constexpr size_t rows = 2, columns = 3, padded_width = 5;
  constexpr size_t stride = padded_width * sizeof(float);
  constexpr std::array<float, rows * columns> packed{1, 4, -2, 3, -8, 6};
  constexpr std::array<float, columns> bias{-2, 1, 4};
  constexpr std::array<float, rows * columns> expected{-1, 5, 2, 1, -7, 10};

  std::array<float, rows * padded_width> padded;
  padded.fill(123.0f);
  upload_projection_rows(padded.data(), stride, packed.data(), rows, columns);
  assert((padded == std::array<float, rows * padded_width>{
      1, 4, -2, 123, 123, 3, -8, 6, 123, 123}));
  std::array<float, rows * columns> output{};
  download_projection_rows(output.data(), padded.data(), stride, rows,
                           columns, nullptr);
  assert(output == packed);
  download_projection_rows(output.data(), padded.data(), stride, rows,
                           columns, bias.data());
  assert(output == expected);
}
