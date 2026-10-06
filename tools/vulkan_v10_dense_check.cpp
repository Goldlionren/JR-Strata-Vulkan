// CPU-only check against the pinned GGML dequantizer, including every dense
// tensor.
#include "ggml-quants.h"
#include "strata/artifact/gguf_reader.hpp"
#include "v10_dense_encoding.hpp"
#include <cmath>
#include <iostream>
int main(int argc, char **argv) {
  if (argc != 2)
    return 2;
  auto model = strata::GgufModel::open(argv[1]);
  size_t tensors = 0, blocks = 0;
  for (size_t sh = 0; sh < model.size(); sh++)
    for (const auto &t : model.shard(sh).tensors()) {
      if (t.type != 14 && t.type != 21 && t.type != 23)
        continue;
      if (t.name.find("_exps") != std::string::npos ||
          t.name == "token_embd.weight")
        continue;
      const size_t stride = t.type == 14 ? 210 : t.type == 21 ? 110 : 136;
      for (size_t sample = 0; sample < 16; sample++) {
        const size_t at = (t.elements() / 256 - 1) * sample / 15;
        const auto *src = model.shard(sh).tensor_data(t) + at * stride;
        float ref[256];
        if (t.type == 14)
          dequantize_row_q6_K(reinterpret_cast<const block_q6_K *>(src), ref,
                              256);
        else if (t.type == 21)
          dequantize_row_iq3_s(reinterpret_cast<const block_iq3_s *>(src), ref,
                               256);
        else
          dequantize_row_iq4_xs(reinterpret_cast<const block_iq4_xs *>(src),
                                ref, 256);
        auto encoded = jr::v10::encode_dense(src, t.type, 256);
        const size_t group = t.type == 14 ? 16 : 32;
        const size_t header = t.type == 14 ? 20 : 12;
        const size_t bits = t.type == 14 ? 6 : 4;
        for (size_t i = 0; i < 256; i++) {
          float scale = strata::fp16_to_fp32(strata::read_u16(encoded.data())) *
                        int8_t(encoded[2 + i / group]);
          const size_t byte = header + i * bits / 8, shift = (i * bits) % 8;
          unsigned word = encoded[byte];
          if (shift + bits > 8)
            word |= unsigned(encoded[byte + 1]) << 8;
          unsigned code = (word >> shift) & ((1u << bits) - 1);
          int symbol;
          if (t.type == 14)
            symbol = code >= 32 ? int(code) - 64 : int(code);
          else if (t.type == 23)
            symbol = kvalues_iq4nl[code];
          else
            symbol = (code & 8 ? -1 : 1) * int(1 + 2 * (code & 7));
          const float actual = scale * symbol;
          if (actual != ref[i]) {
            std::cerr << "mismatch " << t.name << " block=" << at << " i=" << i
                      << " " << actual << " vs " << ref[i] << '\n';
            return 1;
          }
        }
        blocks++;
      }
      tensors++;
    }
  std::cout << "PASS lossless dense encoding: " << tensors << " tensors, "
            << blocks << " real blocks\n";
}
