// Compare actual captured Q/K/V data against the existing batch attention
// kernel.
#include "v10_support.hpp"
#include <bit>
#include <cmath>
using namespace jr::v10;
struct Push {
  uint32_t op = 0, a = 0, b = 0, c = 0, d = 0, e = 0, f = 0, g = 0, h = 0,
           i = 0, j = 0, k = 0, l = 0, m = 0, n = 0, o = 0;
};
int main(int argc, char **argv) {
  try {
    if (argc < 4 || argc > 7)
      throw std::runtime_error("usage: attention-check captured-draft.f32 "
                               "kv.bin query-float-offset [tile.spv] "
                               "[heads-per-group] [current-reference]");
    auto selector = jr::vk::selector_from_env();
    if (selector.vendor_id != 0x8086 || selector.device_id != 0xe211)
      throw std::runtime_error("only JR_VK_DEVICE=8086:e211");
    jr::vk::Context ctx(selector);
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(ctx.physical_device(), &properties);
    std::cout << "SLM limit " << properties.limits.maxComputeSharedMemorySize
              << " B\n";
    if (properties.limits.maxComputeSharedMemorySize < 65536)
      throw std::runtime_error("tile requires 64 KiB SLM");
    constexpr uint32_t Q = 0, SEL = 6144, PART = 10240,
                       OUT = PART + 24 * 65 * 258,
                       STRIDE = (OUT + 6144 + 63) & ~63u;
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
    auto real = read_bytes(argv[1]), rawkv = read_bytes(argv[2]);
    uint32_t qoff = std::stoul(argv[3]);
    if (real.size() < (qoff + 6144) * 4 ||
        rawkv.size() != 2ull * 8192 * 512 * 2)
      throw std::runtime_error("captured layout mismatch");
    // Repeat real captured prefix cells to exercise full-size gathers with
    // nonzero KV.
    for (uint32_t plane = 0; plane < 2; plane++)
      for (uint32_t pos = 74; pos < 8192; pos++)
        std::memcpy(rawkv.data() + (uint64_t(plane) * 8192 + pos) * 1024,
                    rawkv.data() + (uint64_t(plane) * 8192 + pos % 74) * 1024,
                    1024);
    auto kv = upload_device(ctx, rawkv.data(), rawkv.size());
    auto dummy = make(65536, true), control = make(5 * 64, true),
         scratch = make(5ull * STRIDE * 4),
         input = make(5ull * STRIDE * 4, true),
         readback = make(2 * 5ull * STRIDE * 4, true);
    std::memset(dummy.mapped, 0, dummy.size);
    std::memset(input.mapped, 0, input.size);
    std::memset(control.mapped, 0, control.size);
    auto *s = static_cast<float *>(input.mapped);
    const auto *q = reinterpret_cast<const float *>(real.data()) + qoff;
    for (uint32_t row = 0; row < 5; row++)
      for (uint32_t i = 0; i < 6144; i++)
        s[row * STRIDE + Q + i] = q[i] * float(1 + .01 * row);
    std::vector<VkDescriptorBufferInfo> d(12, info(dummy));
    d[1] = info(scratch);
    d[2] = info(control);
    d[4] = info(kv);
    bool current_reference =
        argc == 7 && std::string(argv[6]) == "current-reference";
    uint32_t heads = argc >= 6 ? std::stoul(argv[5]) : 12;
    if (!heads || 12 % heads)
      throw std::runtime_error("invalid head group");
    auto reference = make_pipeline(
             ctx, current_reference ? JR_ATTN_TILE : JR_ATTN_REFERENCE, d,
             sizeof(Push)),
         combine_pipe = make_pipeline(ctx, JR_ATTN_REFERENCE, d, sizeof(Push)),
         tile = make_pipeline(ctx, argc >= 5 ? argv[4] : JR_ATTN_TILE, d,
                              sizeof(Push));
    VkCommandPoolCreateInfo pi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pi.queueFamilyIndex = ctx.queue_family();
    VkCommandPool pool;
    vk_check(vkCreateCommandPool(ctx.device(), &pi, nullptr, &pool), "pool");
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
    auto run = [&](Pipeline &pipe, Push push, uint32_t x, uint32_t y,
                   uint32_t z = 1) {
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.pipeline);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.layout,
                              0, 1, &pipe.set, 0, nullptr);
      vkCmdPushConstants(cmd, pipe.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                         sizeof(Push), &push);
      vkCmdDispatch(cmd, x, y, z);
      barrier();
    };
    uint64_t mismatches = 0;
    for (uint32_t count : {513u, 1024u, 2051u, 4096u, 8192u})
      for (uint32_t rows : {1u, 2u, 3u, 5u}) {
        for (uint32_t row = 0; row < rows; row++) {
          static_cast<uint32_t *>(control.mapped)[row * 16] = count - 1;
          for (uint32_t i = 0; i < std::min(count, 2051u); i++)
            s[row * STRIDE + SEL + i] = std::bit_cast<float>((i * 137) % count);
        }
        VkCommandBufferBeginInfo bi{
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        vk_check(vkBeginCommandBuffer(cmd, &bi), "begin");
        VkBufferCopy cp{0, 0, input.size};
        vkCmdCopyBuffer(cmd, input.buffer, scratch.buffer, 1, &cp);
        barrier();
        vkCmdResetQueryPool(cmd, queries, 0, 4);
        Push p{0, Q, PART, SEL, 0, 8192};
        p.l = STRIDE;
        p.k = 16;
        uint32_t splits = (std::min(count, 2051u) + 31) / 32;
        for (uint32_t variant = 0; variant < 2; variant++) {
          cp = {0, 0, input.size};
          vkCmdCopyBuffer(cmd, input.buffer, scratch.buffer, 1, &cp);
          barrier();
          vkCmdFillBuffer(cmd, scratch.buffer, PART * 4, 24 * 65 * 258 * 4,
                          0x7fc00001u);
          barrier();
          vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                              queries, variant * 2);
          for (uint32_t repeat = 0; repeat < 30; repeat++)
            run(variant ? tile : reference, p,
                variant ? 24 / heads : (current_reference ? 2 : 3), splits,
                rows);
          vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                              queries, variant * 2 + 1);
          Push combine{1, PART, OUT};
          combine.l = STRIDE;
          combine.k = 16;
          run(combine_pipe, combine, 24, rows);
          cp = {0, variant * scratch.size, scratch.size};
          vkCmdCopyBuffer(cmd, scratch.buffer, readback.buffer, 1, &cp);
          barrier();
        }
        VkMemoryBarrier host{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        host.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        host.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
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
        uint64_t stamps[4];
        vk_check(vkGetQueryPoolResults(ctx.device(), queries, 0, 4,
                                       sizeof(stamps), stamps, 8,
                                       VK_QUERY_RESULT_64_BIT),
                 "timestamps");
        auto *ref = static_cast<float *>(readback.mapped),
             *got = ref + 5 * STRIDE;
        uint64_t bad = 0;
        double maximum = 0;
        for (uint32_t row = 0; row < rows; row++)
          for (uint32_t i = 0; i < 6144; i++) {
            uint32_t at = row * STRIDE + OUT + i;
            bad += std::bit_cast<uint32_t>(ref[at]) !=
                   std::bit_cast<uint32_t>(got[at]);
            maximum = std::max(maximum, double(std::abs(ref[at] - got[at])));
          }
        mismatches += bad;
        double scale = properties.limits.timestampPeriod * 1e-6 / 30;
        std::cout << "count=" << count << " rows=" << rows
                  << " differing=" << bad << " max=" << maximum
                  << " reference_ms=" << (stamps[1] - stamps[0]) * scale
                  << " tile_ms=" << (stamps[3] - stamps[2]) * scale << "\n";
        vk_check(vkResetFences(ctx.device(), 1, &fence), "reset fence");
        vk_check(vkResetCommandPool(ctx.device(), pool, 0), "reset pool");
      }
    // Compare complete selected-cell lists, including pooled-block ties and
    // tails.
    std::array<Pipeline, 5> old_select;
    for (uint32_t row = 0; row < 5; row++) {
      auto sd = d;
      sd[1].offset = row * STRIDE * 4;
      sd[1].range = STRIDE * 4;
      sd[2].offset = row * 64;
      sd[2].range = 64;
      old_select[row] =
          make_pipeline(ctx, JR_QSA_REFERENCE_OPS, sd, sizeof(Push));
    }
    auto select = make_pipeline(ctx, JR_QSA_SELECT, d, sizeof(Push));
    for (uint32_t count : {2051u, 2052u, 2053u, 2054u, 2055u, 3072u, 8188u}) {
      for (uint32_t row = 0; row < 5; row++) {
        static_cast<uint32_t *>(control.mapped)[row * 16] = count - 1 + row;
        for (uint32_t i = 0; i < 2054; i++)
          s[row * STRIDE + Q + i] = std::abs(q[i % 512]);
      }
      VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
      vk_check(vkBeginCommandBuffer(cmd, &bi), "select begin");
      VkBufferCopy cp{0, 0, input.size};
      vkCmdCopyBuffer(cmd, input.buffer, scratch.buffer, 1, &cp);
      barrier();
      vkCmdResetQueryPool(cmd, queries, 0, 4);
      vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, queries,
                          0);
      for (uint32_t row = 0; row < 5; row++) {
        uint32_t nk = count + row;
        run(old_select[row], {13, Q, 0, SEL},
            ((nk <= 2051 ? nk : nk / 4 + 1) + 255) / 256, 1);
      }
      vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, queries,
                          1);
      cp = {0, 0, scratch.size};
      vkCmdCopyBuffer(cmd, scratch.buffer, readback.buffer, 1, &cp);
      barrier();
      Push sp{0, Q, 0, SEL};
      sp.l = STRIDE;
      sp.k = 16;
      vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, queries,
                          2);
      uint32_t last = count + 4;
      run(select, sp, last <= 2051 ? (last + 255) / 256 : (last / 4 + 8) / 8,
          5);
      vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, queries,
                          3);
      cp.dstOffset = scratch.size;
      vkCmdCopyBuffer(cmd, scratch.buffer, readback.buffer, 1, &cp);
      VkMemoryBarrier host{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
      host.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
      host.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                           VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &host, 0, nullptr,
                           0, nullptr);
      vk_check(vkEndCommandBuffer(cmd), "select end");
      VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
      si.commandBufferCount = 1;
      si.pCommandBuffers = &cmd;
      vk_check(vkQueueSubmit(ctx.queue(), 1, &si, fence), "select submit");
      vk_check(
          vkWaitForFences(ctx.device(), 1, &fence, VK_TRUE, 120000000000ull),
          "select wait");
      auto *ref = static_cast<float *>(readback.mapped),
           *got = ref + 5 * STRIDE;
      uint64_t bad = 0;
      for (uint32_t row = 0; row < 5; row++)
        for (uint32_t i = 0; i < 2051; i++)
          bad += std::bit_cast<uint32_t>(ref[row * STRIDE + SEL + i]) !=
                 std::bit_cast<uint32_t>(got[row * STRIDE + SEL + i]);
      mismatches += bad;
      uint64_t stamps[4];
      vk_check(vkGetQueryPoolResults(ctx.device(), queries, 0, 4,
                                     sizeof(stamps), stamps, 8,
                                     VK_QUERY_RESULT_64_BIT),
               "select timestamps");
      double scale = properties.limits.timestampPeriod * 1e-6;
      std::cout << "selection count=" << count << " rows=5 differing=" << bad
                << " reference_ms=" << (stamps[1] - stamps[0]) * scale
                << " batch_ms=" << (stamps[3] - stamps[2]) * scale << "\n";
      vk_check(vkResetFences(ctx.device(), 1, &fence), "select reset fence");
      vk_check(vkResetCommandPool(ctx.device(), pool, 0), "select reset pool");
    }
    vkDestroyQueryPool(ctx.device(), queries, nullptr);
    vkDestroyFence(ctx.device(), fence, nullptr);
    vkDestroyCommandPool(ctx.device(), pool, nullptr);
    return mismatches ? 1 : 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << "\n";
    return 2;
  }
}
