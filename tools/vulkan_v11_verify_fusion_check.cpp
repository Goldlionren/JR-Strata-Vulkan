// V11.1 producer/consumer fusion: compare every FP32/Q8/KV bit with RC kernels.
#include "strata/artifact/gguf_reader.hpp"
#include "v10_support.hpp"
#include <bit>
#include <iostream>
using namespace jr::v10;
struct Push {
  uint32_t op = 0, a = 0, b = 0, c = 0, d = 0, e = 0, f = 0, g = 0, h = 0,
           i = 0, j = 0, k = 0, l = 0, m = 0, n = 0, o = 0;
};
#include "vulkan_v11_check_work.hpp"

int main(int argc, char **argv) {
  try {
    if (argc != 3)
      throw std::runtime_error(
          "usage: verify-fusion-check native.gguf captured-base.f32");
    auto sel = jr::vk::selector_from_env();
    if (sel.vendor_id != 0x8086 || sel.device_id != 0xe211)
      throw std::runtime_error("only JR_VK_DEVICE=8086:e211");
    jr::vk::Context ctx(sel);
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(ctx.physical_device(), &props);
    auto model = strata::GgufModel::open(argv[1]);
    Work w;
    auto raw = read_bytes(argv[2]);
    if (raw.size() != w.end * 4ull)
      throw std::runtime_error("capture layout mismatch");
    std::vector<uint8_t> weight_data;
    uint32_t gamma[4];
    const char *names[] = {"blk.0.ssm_norm.weight", "blk.3.attn_q_norm.weight",
                           "blk.3.attn_k_norm.weight",
                           "blk.3.indexer.q_norm.weight"};
    for (uint32_t i = 0; i < 4; i++) {
      size_t shard;
      auto t = model.find(names[i], &shard);
      if (!t || t->type != 0)
        throw std::runtime_error("expected F32 norm");
      gamma[i] = align_up(weight_data.size(), 64);
      weight_data.resize(gamma[i] + t->shape[0] * 4);
      std::memcpy(weight_data.data() + gamma[i],
                  model.shard(shard).tensor_data(*t), t->shape[0] * 4);
      auto f = reinterpret_cast<float *>(weight_data.data() + gamma[i]);
      for (uint32_t j = 0; j < t->shape[0]; j++)
        f[j] += 1;
    }
    auto make = [&](uint64_t size, bool host = false) {
      return make_buffer(ctx, size,
                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                             VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                             VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                         host ? VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
                              : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                         host);
    };
    auto weights = upload_device(ctx, weight_data.data(), weight_data.size()),
         input = make(5ull * w.end * 4, true), control = make(5 * 64, true),
         dummy = make(65536, true),
         readback = make(10ull * w.end * 4 + 2ull * 8192 * 512 * 2 * 2, true);
    std::memset(dummy.mapped, 0, dummy.size);
    std::memset(control.mapped, 0, control.size);
    for (uint32_t t = 0; t < 5; t++) {
      auto dst = static_cast<float *>(input.mapped) + t * w.end;
      std::memcpy(dst, raw.data(), raw.size());
      for (auto [off, n] :
           std::vector<std::pair<uint32_t, uint32_t>>{{w.rec, 6144},
                                                      {w.z, 6144},
                                                      {w.qfull, 12288},
                                                      {w.kcur, 512},
                                                      {w.vcur, 512},
                                                      {w.idxq, 512}})
        for (uint32_t j = 0; j < n; j++)
          dst[off + j] *= float(1 + .01 * t);
    }
    std::array<Buffer, 2> scratch, kv, diag, route, rw;
    std::array<std::array<Pipeline, 6>, 2> pipe;
    for (uint32_t i = 0; i < 512 * 8; i++)
      static_cast<uint32_t *>(dummy.mapped)[i] = i + 1;
    for (uint32_t v = 0; v < 2; v++) {
      scratch[v] = make(input.size);
      kv[v] = make(8192ull * 512 * 2 * 2);
      diag[v] = make(5 * 65536, true);
      route[v] = make(5 * 512, true);
      rw[v] = make(5 * 256, true);
      std::memset(route[v].mapped, 0, route[v].size);
      std::memset(rw[v].mapped, 0, rw[v].size);
      std::memset(diag[v].mapped, 0, diag[v].size);
      std::vector<VkDescriptorBufferInfo> d(12, info(dummy));
      d[0] = info(weights);
      d[1] = info(scratch[v]);
      d[2] = info(control);
      d[4] = info(kv[v]);
      d[5] = info(diag[v]);
      d[8] = info(route[v]);
      d[9] = info(rw[v]);
      pipe[v][0] = make_pipeline(ctx, JR_FUSION_NORM, d, sizeof(Push));
      pipe[v][1] = make_pipeline(ctx, JR_FUSION_QROUND, d, sizeof(Push));
      pipe[v][2] = make_pipeline(ctx, JR_FUSION_QSA_OPS, d, sizeof(Push));
      pipe[v][3] = make_pipeline(ctx, JR_FUSION_GDN, d, sizeof(Push));
      pipe[v][4] = make_pipeline(ctx, JR_FUSION_QSA, d, sizeof(Push));
      pipe[v][5] = make_pipeline(
          ctx, v ? JR_FUSION_ROUTER : JR_FUSION_ROUTER_REF, d, sizeof(Push));
    }
    VkCommandPoolCreateInfo ci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    ci.queueFamilyIndex = ctx.queue_family();
    VkCommandPool pool;
    vk_check(vkCreateCommandPool(ctx.device(), &ci, nullptr, &pool), "pool");
    VkCommandBufferAllocateInfo ai{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = pool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkCommandBuffer cmd;
    vk_check(vkAllocateCommandBuffers(ctx.device(), &ai, &cmd), "command");
    VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence;
    vk_check(vkCreateFence(ctx.device(), &fi, nullptr, &fence), "fence");
    VkQueryPoolCreateInfo qi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    qi.queryType = VK_QUERY_TYPE_TIMESTAMP;
    qi.queryCount = 4;
    VkQueryPool queries;
    vk_check(vkCreateQueryPool(ctx.device(), &qi, nullptr, &queries),
             "queries");
    auto barrier = [&]() {
      VkMemoryBarrier b{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
      b.srcAccessMask =
          VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
      b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                        VK_ACCESS_TRANSFER_READ_BIT;
      vkCmdPipelineBarrier(
          cmd,
          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
          0, 1, &b, 0, nullptr, 0, nullptr);
    };
    auto run = [&](Pipeline &p, Push push, uint32_t x, uint32_t y) {
      push.l = w.end;
      if (!push.k)
        push.k = 16;
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p.pipeline);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p.layout, 0,
                              1, &p.set, 0, nullptr);
      vkCmdPushConstants(cmd, p.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                         sizeof(push), &push);
      vkCmdDispatch(cmd, x, y, 1);
      barrier();
    };
    uint64_t bad_total = 0;
    for (uint32_t base : {73u, 511u, 513u, 2051u, 8187u})
      for (uint32_t n = 1; n <= 5; n++)
        for (uint32_t mode = 0; mode < 5; mode++) {
          for (uint32_t t = 0; t < 5; t++) {
            static_cast<uint32_t *>(control.mapped)[t * 16] = base + t;
            if (mode == 4)
              for (uint32_t i = 0; i < 512; i++) {
                float value =
                    reinterpret_cast<const float *>(raw.data())[w.scores + i] *
                    float(1 + .01 * t);
                if (base == 511)
                  value = 0;
                if (base == 513)
                  value = float(int(i % 7) - 3);
                if (base == 2051)
                  value = -64 + 128 * float(i) / 511;
                static_cast<float *>(input.mapped)[t * w.end + w.scores + i] =
                    value;
              }
          }
          VkCommandBufferBeginInfo bi{
              VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
          vk_check(vkBeginCommandBuffer(cmd, &bi), "begin");
          vkCmdResetQueryPool(cmd, queries, 0, 4);
          for (uint32_t v = 0; v < 2; v++) {
            VkBufferCopy cp{0, 0, input.size};
            vkCmdCopyBuffer(cmd, input.buffer, scratch[v].buffer, 1, &cp);
            vkCmdFillBuffer(cmd, kv[v].buffer, 0, kv[v].size, 0);
            barrier();
            vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                queries, v * 2);
            for (uint32_t repeat = 0; repeat < 20; repeat++) {
              if (mode == 0) {
                Push p{7, w.rec, w.y, gamma[0], w.z};
                if (v) {
                  p.f = w.round;
                  p.n = 1024;
                  run(pipe[v][3], p, 48, n);
                } else {
                  run(pipe[v][0], p, 48, n);
                  p = {0, w.y, w.round, 6144};
                  p.n = 1024;
                  run(pipe[v][1], p, 192, n);
                }
              } else if (mode == 4) {
                Push p{0, w.scores, 16};
                p.d = 65536 / 4;
                p.m = 64;
                p.k = 128;
                run(pipe[v][5], p, 1, n);
              } else {
                uint32_t width = mode == 3 ? 128 : 256,
                         heads = mode == 1   ? 24
                                 : mode == 2 ? 2
                                             : 4,
                         off = mode == 1   ? w.q
                               : mode == 2 ? w.kcur
                                           : w.idxq;
                Push p;
                if (v) {
                  p = {mode == 1   ? 0u
                       : mode == 2 ? 1u
                                   : 2u,
                       mode == 1 ? w.qfull : off,
                       off,
                       gamma[mode],
                       width,
                       w.vcur,
                       8192};
                  run(pipe[v][4], p, heads, n);
                } else {
                  if (mode == 1)
                    run(pipe[v][2], {8, w.qfull, w.q}, 24, n);
                  run(pipe[v][0], {1, off, off, gamma[mode], width, heads, 1},
                      heads, n);
                  run(pipe[v][2], {9, off, width, heads},
                      (heads * 32 + 255) / 256, n);
                  if (mode == 2)
                    run(pipe[v][2], {10, w.kcur, w.vcur, 0, 8192}, 1, n);
                }
              }
            }
            vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                queries, v * 2 + 1);
            barrier();
            cp = {0, v * input.size, input.size};
            vkCmdCopyBuffer(cmd, scratch[v].buffer, readback.buffer, 1, &cp);
            cp = {0, 2 * input.size + v * kv[v].size, kv[v].size};
            vkCmdCopyBuffer(cmd, kv[v].buffer, readback.buffer, 1, &cp);
            barrier();
          }
          VkMemoryBarrier host{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
          host.srcAccessMask =
              VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT;
          host.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
          vkCmdPipelineBarrier(cmd,
                               VK_PIPELINE_STAGE_TRANSFER_BIT |
                                   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                               VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &host, 0,
                               nullptr, 0, nullptr);
          vk_check(vkEndCommandBuffer(cmd), "end");
          VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
          si.commandBufferCount = 1;
          si.pCommandBuffers = &cmd;
          vk_check(vkQueueSubmit(ctx.queue(), 1, &si, fence), "submit");
          vk_check(vkWaitForFences(ctx.device(), 1, &fence, VK_TRUE,
                                   120000000000ull),
                   "wait");
          auto a = static_cast<uint32_t *>(readback.mapped),
               b = a + input.size / 4;
          uint64_t bad = 0;
          for (uint32_t i = 0; i < input.size / 4; i++)
            bad += a[i] != b[i];
          a += 2 * input.size / 4;
          b = a + kv[0].size / 4;
          for (uint32_t i = 0; i < kv[0].size / 4; i++)
            bad += a[i] != b[i];
          bad +=
              std::memcmp(route[0].mapped, route[1].mapped, route[0].size) != 0;
          bad += std::memcmp(rw[0].mapped, rw[1].mapped, rw[0].size) != 0;
          bad += std::memcmp(diag[0].mapped, diag[1].mapped, diag[0].size) != 0;
          bad_total += bad;
          uint64_t ts[4];
          vk_check(vkGetQueryPoolResults(ctx.device(), queries, 0, 4,
                                         sizeof(ts), ts, 8,
                                         VK_QUERY_RESULT_64_BIT),
                   "times");
          double scale = props.limits.timestampPeriod * 1e-6 / 20;
          std::cout << "base=" << base << " T=" << n << " mode=" << mode
                    << " differing=" << bad
                    << " reference_ms=" << (ts[1] - ts[0]) * scale
                    << " fused_ms=" << (ts[3] - ts[2]) * scale << "\n";
          if (bad)
            return 1;
          vk_check(vkResetFences(ctx.device(), 1, &fence), "reset fence");
          vk_check(vkResetCommandPool(ctx.device(), pool, 0), "reset pool");
        }
    vkDestroyQueryPool(ctx.device(), queries, nullptr);
    vkDestroyFence(ctx.device(), fence, nullptr);
    vkDestroyCommandPool(ctx.device(), pool, nullptr);
    return bad_total ? 1 : 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << "\n";
    return 2;
  }
}
