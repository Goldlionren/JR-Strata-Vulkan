#pragma once
#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"
#include "strata/artifact/dequant.hpp"
#include <cstring>
#include <stdexcept>
#include <vector>

namespace jr::v10 {
// Private lossless dense layouts. Retain the original FP16 block scale and
// signed integer multipliers; pack logical Q6 symbols in six bits and IQ3/IQ4
// symbols in four bits. No expanded byte weights or F32 scale arrays persist.
inline std::vector<uint8_t> encode_dense(const uint8_t *src, uint32_t type,
                                         uint64_t elements) {
  if ((type != 14 && type != 21 && type != 23) || elements % 256)
    throw std::runtime_error("invalid compact dense input");
  const uint32_t stride = type == 14 ? 212 : 140;
  const uint32_t header = type == 14 ? 20 : 12;
  const uint32_t bits = type == 14 ? 6 : 4;
  std::vector<uint8_t> result(elements / 256 * stride, 0);
  for (uint64_t block_index = 0; block_index < elements / 256; block_index++) {
    const auto *block = src + block_index * (type == 14   ? 210
                                             : type == 23 ? 136
                                                          : 110);
    auto *out = result.data() + block_index * stride;
    std::memcpy(out, block + (type == 14 ? 208 : 0), 2);
    for (uint32_t g = 0; g < (type == 14 ? 16u : 8u); g++) {
      int scale;
      if (type == 14)
        scale = int8_t(block[192 + g]);
      else if (type == 23)
        scale = int(((block[4 + g / 2] >> ((g % 2) * 4)) & 15) |
                    (((strata::read_u16(block + 2) >> (g * 2)) & 3) << 4)) -
                32;
      else
        scale = 1 + 2 * ((block[106 + g / 2] >> ((g % 2) * 4)) & 15);
      out[2 + g] = uint8_t(int8_t(scale));
    }
    for (uint32_t j = 0; j < 256; j++) {
      uint32_t code;
      if (type == 14) {
        uint32_t half = j / 128, z = j % 128, l = z % 32, g = z / 32;
        int symbol =
            int(((block[half * 64 + (g % 2) * 32 + l] >> ((g / 2) * 4)) & 15) |
                (((block[128 + half * 32 + l] >> (g * 2)) & 3) << 4)) -
            32;
        code = uint32_t(symbol) & 63;
      } else if (type == 23)
        code =
            (block[8 + (j / 32) * 16 + j % 16] >> (((j % 32) / 16) * 4)) & 15;
      else {
        uint32_t g = j / 32, pair = (j % 32) / 8, part = (j % 8) / 4;
        uint32_t index = block[2 + g * 8 + pair * 2 + part] |
                         (((block[66 + g] >> (pair * 2 + part)) & 1) << 8);
        uint32_t magnitude = (iq3s_grid[index] >> ((j % 4) * 8)) & 255;
        if (!(magnitude & 1) || magnitude > 15)
          throw std::runtime_error("unexpected IQ3_S symbol");
        code = (magnitude / 2) |
               (((block[74 + g * 4 + pair] >> (j % 8)) & 1) << 3);
      }
      uint32_t byte = header + j * bits / 8, shift = (j * bits) % 8;
      out[byte] |= uint8_t(code << shift);
      if (shift + bits > 8)
        out[byte + 1] |= uint8_t(code >> (8 - shift));
    }
  }
  return result;
}
} // namespace jr::v10
