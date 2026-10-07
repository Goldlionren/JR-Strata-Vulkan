// Compare actual recurrent inputs, final state, every live prefix and flags.
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
    if (argc < 3 || argc > 4)
      throw std::runtime_error("usage: recurrent-check captured-base.f32 "
                               "candidate.spv [column groups]");
    auto sel = jr::vk::selector_from_env();
    if (sel.vendor_id != 0x8086 || sel.device_id != 0xe211)
      throw std::runtime_error("only JR_VK_DEVICE=8086:e211");
    jr::vk::Context ctx(sel);
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(ctx.physical_device(), &props);
    uint32_t groups = argc == 4 ? std::stoul(argv[3]) : 32,
             width = 4096 / groups;
    if (width > props.limits.maxComputeWorkGroupInvocations)
      throw std::runtime_error("unsupported workgroup size");
    Work w;
    auto raw = read_bytes(argv[1]);
    if (raw.size() != w.end * 4ull)
      throw std::runtime_error("capture layout mismatch");
    constexpr uint32_t STATE = 128 * 48 * 128;
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
         state_readback = make(STATE * 4ull * 10, true);
    std::memset(dummy.mapped, 0, dummy.size);
    for (uint32_t t = 0; t < 5; t++) {
      auto dst = static_cast<float *>(input.mapped) + t * w.end;
      std::memcpy(dst, raw.data(), raw.size());
      for (uint32_t i = 0; i < 10240; i++)
        dst[w.h + i] *= float(1 + .001 * t);
    }
    std::vector<float> initial(STATE);
    for (uint32_t i = 0; i < STATE; i++)
      initial[i] =
          reinterpret_cast<const float *>(raw.data())[i % 10240] * .0001f +
          std::sin(float(i) * .013f) * .0001f;
    auto initial_state = upload_device(ctx, initial.data(), initial.size() * 4);
    std::array<Buffer, 2> scratch, state, snapshots, diag;
    std::array<Pipeline, 2> pipe;
    for (uint32_t v = 0; v < 2; v++) {
      scratch[v] = make(input.size);
      state[v] = make(STATE * 4ull);
      snapshots[v] = make(STATE * 4ull * 4);
      diag[v] = make(65536, true);
      std::vector<VkDescriptorBufferInfo> d(12, info(dummy));
      d[1] = info(scratch[v]);
      d[3] = info(state[v]);
      d[5] = info(diag[v]);
      d[10] = info(snapshots[v]);
      pipe[v] = make_pipeline(ctx, v ? argv[2] : JR_RECURRENT_REFERENCE, d,
                              sizeof(Push));
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
    for (uint32_t n = 1; n <= 5; n++)
      for (uint32_t fixture : {0u, 0x7fc12345u, 0x7f800000u, 0xff800000u}) {
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
          barrier(); // Complete the copy before filling its first word.
          if (fixture)
            vkCmdFillBuffer(cmd, state[v].buffer, 0, 4, fixture);
          vkCmdFillBuffer(cmd, snapshots[v].buffer, 0, snapshots[v].size,
                          0x7fc00001u);
          barrier();
          auto &p = pipe[v];
          Push push{0, w.h, w.rec, w.alpha, w.beta};
          push.i = STATE;
          push.l = w.end;
          push.n = 1024;
          push.o = n;
          vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p.pipeline);
          vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p.layout,
                                  0, 1, &p.set, 0, nullptr);
          vkCmdPushConstants(cmd, p.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                             sizeof(push), &push);
          vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                              queries, v * 2);
          for (uint32_t r = 0; r < 20; r++) {
            vkCmdDispatch(cmd, v ? groups : 32, 48, 1);
            barrier();
          }
          vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                              queries, v * 2 + 1);
          barrier();
          cp = {0, v * input.size, input.size};
          vkCmdCopyBuffer(cmd, scratch[v].buffer, scratch_readback.buffer, 1,
                          &cp);
          cp = {0, v * 5ull * STATE * 4, state[v].size};
          vkCmdCopyBuffer(cmd, state[v].buffer, state_readback.buffer, 1, &cp);
          cp = {0, (v * 5ull + 1) * STATE * 4, snapshots[v].size};
          vkCmdCopyBuffer(cmd, snapshots[v].buffer, state_readback.buffer, 1,
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
        vk_check(
            vkWaitForFences(ctx.device(), 1, &fence, VK_TRUE, 120000000000ull),
            "wait");
        uint64_t bad = 0;
        auto a = static_cast<uint32_t *>(scratch_readback.mapped),
             b = a + input.size / 4;
        for (uint32_t i = 0; i < input.size / 4; i++)
          bad += a[i] != b[i];
        a = static_cast<uint32_t *>(state_readback.mapped);
        b = a + 5 * STATE;
        uint32_t live = std::min(4u, n - 1);
        for (uint32_t i = 0; i < (1 + live) * STATE; i++)
          bad += a[i] != b[i];
        bad += std::memcmp(diag[0].mapped, diag[1].mapped, diag[0].size) != 0;
        auto reference_flags = static_cast<uint32_t *>(diag[0].mapped);
        if ((reference_flags[1027] != 0) != (fixture != 0))
          throw std::runtime_error(
              "reference finite injection was not observed");
        auto flags = static_cast<uint32_t *>(diag[1].mapped);
        if ((flags[1027] != 0) != (fixture != 0))
          throw std::runtime_error(
              "finite guard did not preserve injected failure");
        uint64_t ts[4];
        vk_check(vkGetQueryPoolResults(ctx.device(), queries, 0, 4, sizeof(ts),
                                       ts, 8, VK_QUERY_RESULT_64_BIT),
                 "times");
        double scale = props.limits.timestampPeriod * 1e-6 / 20;
        std::cout << "N=" << n << " fixture=" << fixture << " differing=" << bad
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
