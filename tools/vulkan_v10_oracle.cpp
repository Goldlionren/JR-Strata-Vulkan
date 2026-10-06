// CPU-only pinned llama.cpp oracle; never initializes a GPU backend.
#include "llama.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>
int main(int argc, char **argv) {
  if (argc < 4) {
    std::fprintf(stderr, "usage: oracle model.gguf tokens.i32 logits.f32\n");
    return 1;
  }
  llama_backend_init();
  auto mp = llama_model_default_params();
  mp.n_gpu_layers = 0;
  mp.load_mode = LLAMA_LOAD_MODE_MMAP;
  mp.load_mtp = false;
  auto *model = llama_model_load_from_file(argv[1], mp);
  if (!model)
    return 2;
  auto cp = llama_context_default_params();
  cp.n_ctx = 8192;
  cp.n_batch = 1;
  cp.n_ubatch = 1;
  cp.n_threads = 8;
  cp.n_threads_batch = 8;
  cp.offload_kqv = false;
  cp.op_offload = false;
  cp.type_k = GGML_TYPE_F16;
  cp.type_v = GGML_TYPE_F16;
  auto *ctx = llama_init_from_model(model, cp);
  if (!ctx)
    return 3;
  std::ifstream file(argv[2], std::ios::binary);
  llama_token token;
  while (file.read(reinterpret_cast<char *>(&token), 4)) {
    auto batch = llama_batch_get_one(&token, 1);
    if (llama_decode(ctx, batch))
      return 4;
  }
  auto *logits = llama_get_logits_ith(ctx, -1);
  int nv = llama_vocab_n_tokens(llama_model_get_vocab(model));
  std::ofstream out(argv[3], std::ios::binary);
  out.write(reinterpret_cast<char *>(logits), nv * 4);
  int best = std::max_element(logits, logits + nv) - logits;
  std::printf("CPU oracle token=%d logit=%.9g\n", best, logits[best]);
  llama_free(ctx);
  llama_model_free(model);
  llama_backend_free();
}
