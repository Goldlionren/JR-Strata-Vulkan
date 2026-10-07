// Compare causal indexer progression and every score bit against the RC.
#include "strata/artifact/gguf_reader.hpp"
#include "v10_support.hpp"
#include "vulkan_v11_check_work.hpp"
#include <bit>
#include <cmath>
#include <iostream>
using namespace jr::v10;
struct Push {
  uint32_t op = 0, a = 0, b = 0, c = 0, d = 0, e = 0, f = 0, g = 0, h = 0,
           i = 0, j = 0, k = 0, l = 0, m = 0, n = 0, o = 0;
};
int main(int argc, char **argv) {
  try {
    if (argc != 4)
      throw std::runtime_error(
          "usage: qsa-score-check native.gguf captured-base.f32 "
          "candidate.spv");
    auto sel = jr::vk::selector_from_env();
    if (sel.vendor_id != 0x8086 || sel.device_id != 0xe211)
      throw std::runtime_error("only JR_VK_DEVICE=8086:e211");
    jr::vk::Context ctx(sel);
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(ctx.physical_device(), &props);
    Work w;
    auto raw = read_bytes(argv[2]);
    if (raw.size() != w.end * 4ull)
      throw std::runtime_error("capture layout mismatch");
    auto model = strata::GgufModel::open(argv[1]);
    std::vector<uint8_t> weight_data;
    const char *name = "blk.3.indexer.k_norm.weight";
    size_t shard;
    auto t = model.find(name, &shard);
    if (!t || t->type != 0 || t->shape != std::vector<uint64_t>{128})
      throw std::runtime_error("expected F32 indexer norm");
    weight_data.resize(128 * 4);
    std::memcpy(weight_data.data(), model.shard(shard).tensor_data(*t),
                weight_data.size());
    for (uint32_t i = 0; i < 128; i++)
      reinterpret_cast<float *>(weight_data.data())[i] += 1;
    auto weights = upload_device(ctx, weight_data.data(), weight_data.size());
    constexpr uint32_t STATE = 512 + 2050 * 128;
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
    auto input = make(5ull * w.end * 4, true), dummy = make(65536, true),
         scratch_readback = make(input.size * 2, true),
         state_readback = make(STATE * 4ull * 2, true),
         control = make(5 * 64, true);
    std::memset(dummy.mapped, 0, dummy.size);
    for (uint32_t t = 0; t < 5; t++) {
      auto dst = static_cast<float *>(input.mapped) + t * w.end;
      std::memcpy(dst, raw.data(), raw.size());
      for (uint32_t i = 0; i < 10240; i++)
        dst[w.h + i] *= float(1 + .001 * t);
      for (uint32_t i = 0; i < 512; i++)
        dst[w.idxq + i] *= float(1 + .01 * t);
      for (uint32_t i = 0; i < 128; i++)
        dst[w.idxraw + i] *= float(1 + .01 * t);
    }
    std::vector<float> initial(STATE);
    for (uint32_t i = 0; i < STATE; i++)
      initial[i] =
          reinterpret_cast<const float *>(raw.data())[i % 10240] * .0001f +
          std::sin(float(i) * .013f) * .0001f;
    auto initial_state = upload_device(ctx, initial.data(), initial.size() * 4);
    std::array<Buffer, 2> scratch, state, diag;
    std::array<std::array<Pipeline, 5>, 2> ordinary;
    std::array<Pipeline, 2> batch;
    for (uint32_t v = 0; v < 2; v++) {
      scratch[v] = make(input.size);
      state[v] = make(STATE * 4ull);
      diag[v] = make(65536, true);
      std::vector<VkDescriptorBufferInfo> d(12, info(dummy));
      d[0] = info(weights);
      d[1] = info(scratch[v]);
      d[3] = info(state[v]);
      d[5] = info(diag[v]);
      d[2] = info(control);
      batch[v] = make_pipeline(ctx, argv[3], d, sizeof(Push), 32);
      for (uint32_t row = 0; row < 5; row++) {
        auto sd = d;
        sd[1].offset = row * w.end * 4ull;
        sd[1].range = w.end * 4ull;
        sd[2].offset = row * 64;
        sd[2].range = 64;
        ordinary[v][row] =
            make_pipeline(ctx, JR_SCORE_REFERENCE, sd, sizeof(Push), 32);
      }
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
                        VK_ACCESS_TRANSFER_READ_BIT |
                        VK_ACCESS_TRANSFER_WRITE_BIT;
      vkCmdPipelineBarrier(
          cmd,
          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
          0, 1, &b, 0, nullptr, 0, nullptr);
    };
    auto run = [&](Pipeline &p, Push push, uint32_t x, uint32_t y = 1) {
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p.pipeline);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p.layout, 0,
                              1, &p.set, 0, nullptr);
      vkCmdPushConstants(cmd, p.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                         sizeof(push), &push);
      vkCmdDispatch(cmd, x, y, 1);
      barrier();
    };
    for (uint32_t base :
         {0u, 73u, 2048u, 2049u, 2050u, 2051u, 2052u, 2053u, 3071u, 8187u})
      for (uint32_t n = 1; n <= 5; n++)
        for (uint32_t repeats : {1u, 20u}) {
          for (uint32_t row = 0; row < 5; row++)
            static_cast<uint32_t *>(control.mapped)[row * 16] = base + row;
          VkCommandBufferBeginInfo bi{
              VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
          vk_check(vkBeginCommandBuffer(cmd, &bi), "begin");
          vkCmdResetQueryPool(cmd, queries, 0, 4);
          for (uint32_t v = 0; v < 2; v++) {
            std::memset(diag[v].mapped, 0, diag[v].size);
            VkBufferCopy cp{0, 0, input.size};
            vkCmdCopyBuffer(cmd, input.buffer, scratch[v].buffer, 1, &cp);
            cp.size = state[v].size;
            vkCmdCopyBuffer(cmd, initial_state.buffer, state[v].buffer, 1, &cp);
            barrier();
            vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                queries, v * 2);
            for (uint32_t r = 0; r < repeats; r++) {
              for (uint32_t row = 0; row < n; row++) {
                run(ordinary[v][row], {11, w.idxraw, 0, 0, 512, 384}, 1);
                uint32_t count = base + row + 1;
                if (!v)
                  run(ordinary[v][row], {12, w.idxq, 512, 384, w.scores},
                      count > 2051 ? count / 4 + 1 : 0);
              }
              if (v) {
                Push score{12, w.idxq, 512, 384, w.scores};
                score.l = w.end;
                score.k = 16;
                uint32_t last = base + n;
                run(batch[v], score, last > 2051 ? last / 4 + 1 : 0, n);
              }
            }
            vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                queries, v * 2 + 1);
            barrier();
            cp = {0, v * input.size, input.size};
            vkCmdCopyBuffer(cmd, scratch[v].buffer, scratch_readback.buffer, 1,
                            &cp);
            cp = {0, v * STATE * 4ull, state[v].size};
            vkCmdCopyBuffer(cmd, state[v].buffer, state_readback.buffer, 1,
                            &cp);
            barrier();
          }
          VkMemoryBarrier host{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
          host.srcAccessMask =
              VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
          host.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
          vkCmdPipelineBarrier(cmd,
                               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                                   VK_PIPELINE_STAGE_TRANSFER_BIT,
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
          uint64_t bad = 0;
          auto a = static_cast<uint32_t *>(scratch_readback.mapped),
               b = a + input.size / 4;
          for (uint32_t i = 0; i < input.size / 4; i++)
            bad += a[i] != b[i];
          a = static_cast<uint32_t *>(state_readback.mapped);
          b = a + STATE;
          for (uint32_t i = 0; i < STATE; i++)
            bad += a[i] != b[i];
          bad += std::memcmp(diag[0].mapped, diag[1].mapped, diag[0].size) != 0;
          uint64_t ts[4];
          vk_check(vkGetQueryPoolResults(ctx.device(), queries, 0, 4,
                                         sizeof(ts), ts, 8,
                                         VK_QUERY_RESULT_64_BIT),
                   "times");
          double scale = props.limits.timestampPeriod * 1e-6 / repeats;
          std::cout << "base=" << base << " N=" << n << " repeats=" << repeats
                    << " differing=" << bad
                    << " rc_ms=" << (ts[1] - ts[0]) * scale
                    << " candidate_ms=" << (ts[3] - ts[2]) * scale << std::endl;
          if (bad)
            return 1;
          vk_check(vkResetFences(ctx.device(), 1, &fence), "reset fence");
          vk_check(vkResetCommandPool(ctx.device(), pool, 0), "reset pool");
        }
    vkDestroyQueryPool(ctx.device(), queries, nullptr);
    vkDestroyFence(ctx.device(), fence, nullptr);
    vkDestroyCommandPool(ctx.device(), pool, nullptr);
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return 2;
  }
}
