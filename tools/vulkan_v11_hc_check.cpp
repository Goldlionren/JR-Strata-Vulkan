// Real-weight Vulkan check: batch HC must preserve every V10 output bit.
#include "strata/artifact/gguf_reader.hpp"
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
    if (argc < 3 || argc > 5)
      throw std::runtime_error("usage: hc-check native.gguf captured-base.f32 "
                               "[candidate down SPV] [candidate up SPV]");
    auto selector = jr::vk::selector_from_env();
    if (selector.vendor_id != 0x8086 || selector.device_id != 0xe211)
      throw std::runtime_error("only JR_VK_DEVICE=8086:e211");
    jr::vk::Context ctx(selector);
    auto model = strata::GgufModel::open(argv[1]);
    std::vector<uint8_t> dense;
    std::array<uint32_t, 4> offsets{};
    const char *roles[] = {"norm", "down", "up", "inject"};
    for (uint32_t i = 0; i < 4; i++) {
      size_t shard;
      auto t = model.find(std::string("blk.0.hc_attn_") + roles[i] + ".weight",
                          &shard);
      if (!t)
        throw std::runtime_error("missing HC weight");
      offsets[i] = align_up(dense.size(), 64);
      dense.resize(offsets[i] + strata::tensor_payload_bytes(*t));
      std::memcpy(dense.data() + offsets[i], model.shard(shard).tensor_data(*t),
                  strata::tensor_payload_bytes(*t));
    }
    constexpr uint32_t E = 2560, HC = 10240, R = 0, XN = HC, LO = 2 * HC,
                       INJ = LO + 320, MIXED = INJ + 64, BLOCK = MIXED + E,
                       ROUND = BLOCK + E,
                       STRIDE = (ROUND + E / 32 * 9 + 63) & ~63u;
    uint32_t bf = 30;
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
    auto weights = upload_device(ctx, dense.data(), dense.size());
    auto dummy = make(65536, true);
    std::memset(dummy.mapped, 0, 65536);
    auto base = make(STRIDE * 6 * 4), batch = make(STRIDE * 6 * 4),
         input = make(STRIDE * 6 * 4, true),
         readback = make(STRIDE * 6 * 8, true);
    auto raw = read_bytes(argv[2]);
    if (raw.size() < HC * 4)
      throw std::runtime_error("missing real residual");
    const float *real = reinterpret_cast<const float *>(raw.data());
    auto *s = static_cast<float *>(input.mapped);
    std::memset(s, 0, input.size);
    for (uint32_t row = 0; row < 6; row++) {
      for (uint32_t i = 0; i < HC; i++)
        s[row * STRIDE + R + i] = real[i] * float(1 + 0.01 * row);
      for (uint32_t i = 0; i < E; i++)
        s[row * STRIDE + BLOCK + i] = std::sin(float(i) * .1f) * .02f;
      for (uint32_t c = 0; c < 4; c++)
        s[row * STRIDE + INJ + c] = float(c) * .5f;
    }
    auto desc = [&](Buffer &scratch, uint32_t row, bool sliced) {
      std::vector<VkDescriptorBufferInfo> d(12, info(dummy));
      d[0] = info(weights);
      d[1] = info(scratch, sliced ? STRIDE * 4 : scratch.size);
      d[1].offset = row * STRIDE * 4;
      return d;
    };
    std::array<Pipeline, 5> ops, small, hc, batch_ops;
    for (uint32_t row = 0; row < 5; row++) {
      auto d = desc(base, row, true);
      ops[row] = make_pipeline(ctx, JR_HC_OPS, d, sizeof(Push));
      small[row] = make_pipeline(ctx, JR_HC_SMALL, d, sizeof(Push), 32, &bf);
      hc[row] = make_pipeline(ctx, JR_HC_UP, d, sizeof(Push));
      batch_ops[row] =
          make_pipeline(ctx, JR_HC_OPS, desc(batch, row, true), sizeof(Push));
    }
    auto d = desc(batch, 0, false);
    auto bn = make_pipeline(ctx, JR_HC_BATCH_NORM, d, sizeof(Push));
    std::array<Pipeline, 6> bd, bu, bd_reference, bu_reference;
    auto qr_reference =
        make_pipeline(ctx, JR_HC_QROUND, desc(base, 0, false), sizeof(Push));
    auto qr_candidate = make_pipeline(ctx, JR_HC_QROUND, d, sizeof(Push));
    for (uint32_t n = 1; n <= 5; n++) {
      uint32_t format = bf | (n << 8);
      bd[n] = make_pipeline(ctx, argc >= 4 ? argv[3] : JR_HC_BATCH_DOWN, d,
                            sizeof(Push), 32, &format);
      bd_reference[n] =
          make_pipeline(ctx, JR_HC_BATCH_DOWN_REFERENCE, desc(base, 0, false),
                        sizeof(Push), 32, &format);
      bu[n] = make_pipeline(ctx, argc == 5 ? argv[4] : JR_HC_BATCH_UP, d,
                            sizeof(Push), 32, &n);
      bu_reference[n] =
          make_pipeline(ctx, JR_HC_BATCH_UP_REFERENCE, desc(base, 0, false),
                        sizeof(Push), 32, &n);
    }
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
    VkPhysicalDeviceProperties properties;
    vkGetPhysicalDeviceProperties(ctx.physical_device(), &properties);
    VkQueryPoolCreateInfo qi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    qi.queryType = VK_QUERY_TYPE_TIMESTAMP;
    qi.queryCount = 8;
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
    auto run = [&](Pipeline &p, Push push, uint32_t x, uint32_t y = 1) {
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p.pipeline);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p.layout, 0,
                              1, &p.set, 0, nullptr);
      vkCmdPushConstants(cmd, p.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                         sizeof(Push), &push);
      vkCmdDispatch(cmd, x, y, 1);
      barrier();
    };
    uint64_t mismatches = 0;
    for (uint32_t n = 1; n <= 5; n++) {
      VkCommandBufferBeginInfo begin{
          VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
      vk_check(vkBeginCommandBuffer(cmd, &begin), "begin");
      vkCmdResetQueryPool(cmd, queries, 0, 8);
      VkBufferCopy cp{0, 0, input.size};
      vkCmdCopyBuffer(cmd, input.buffer, base.buffer, 1, &cp);
      vkCmdCopyBuffer(cmd, input.buffer, batch.buffer, 1, &cp);
      barrier();
      Push np{25, R, XN, offsets[0], E, 4, 2, 0, BLOCK, INJ},
          dp{1, offsets[1], XN, LO, HC, 324, 30, HC * 2, 0, offsets[3], INJ},
          hp{0, offsets[2], LO, MIXED, XN};
      for (uint32_t row = 0; row < n; row++) {
        run(ops[row], np, 4);
        run(small[row], dp, 324);
        run(hc[row], hp, E / 4);
      }
      np.l = dp.l = hp.l = STRIDE;
      hp.f = ROUND;
      Push qp{0, MIXED, ROUND, E};
      qp.l = STRIDE;
      qp.o = n;
      dp.n = hp.n = qp.n = 1024;
      np.o = dp.o = hp.o = n;
      vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, queries,
                          0);
      for (uint32_t repeat = 0; repeat < 40; repeat++)
        run(bd_reference[n], dp, 324);
      vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, queries,
                          1);
      vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, queries,
                          4);
      for (uint32_t repeat = 0; repeat < 40; repeat++) {
        run(bu_reference[n], hp, E / 4);
        run(qr_reference, qp, E / 32, n);
      }
      vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, queries,
                          5);
      run(bn, np, 4, n);
      run(bd[n], dp, 324);
      run(bu[n], hp, argc == 5 ? E / 32 : E / 4);
      if (argc != 5)
        run(qr_candidate, qp, E / 32, n);
      vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, queries,
                          2);
      for (uint32_t repeat = 0; repeat < 40; repeat++)
        run(bd[n], dp, 324);
      vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, queries,
                          3);
      vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, queries,
                          6);
      for (uint32_t repeat = 0; repeat < 40; repeat++) {
        run(bu[n], hp, argc == 5 ? E / 32 : E / 4);
        if (argc != 5)
          run(qr_candidate, qp, E / 32, n);
      }
      vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, queries,
                          7);
      cp = {0, 0, base.size};
      vkCmdCopyBuffer(cmd, base.buffer, readback.buffer, 1, &cp);
      cp.dstOffset = base.size;
      vkCmdCopyBuffer(cmd, batch.buffer, readback.buffer, 1, &cp);
      VkMemoryBarrier host{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
      host.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
      host.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                           VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &host, 0, nullptr,
                           0, nullptr);
      vk_check(vkEndCommandBuffer(cmd), "end");
      VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
      si.commandBufferCount = 1;
      si.pCommandBuffers = &cmd;
      vk_check(vkQueueSubmit(ctx.queue(), 1, &si, fence), "submit");
      vk_check(
          vkWaitForFences(ctx.device(), 1, &fence, VK_TRUE, 120000000000ull),
          "wait");
      uint64_t stamps[8];
      vk_check(vkGetQueryPoolResults(ctx.device(), queries, 0, 8,
                                     sizeof(stamps), stamps, 8,
                                     VK_QUERY_RESULT_64_BIT),
               "timestamps");
      double scale = properties.limits.timestampPeriod * 1e-6 / 40;
      std::cout << "T=" << n
                << " RC_down_ms=" << (stamps[1] - stamps[0]) * scale
                << " candidate_down_ms=" << (stamps[3] - stamps[2]) * scale
                << "\n";
      std::cout << "T=" << n
                << " RC_up_round_ms=" << (stamps[5] - stamps[4]) * scale
                << " candidate_up_round_ms=" << (stamps[7] - stamps[6]) * scale
                << "\n";
      auto *ref = static_cast<float *>(readback.mapped),
           *got = ref + STRIDE * 6;
      for (auto [name, start, count] :
           std::vector<std::tuple<const char *, uint32_t, uint32_t>>{
               {"residual", R, HC},
               {"norm", XN, HC},
               {"down", LO, 320},
               {"inject", INJ, 4},
               {"mixed", MIXED, E},
               {"Q8", ROUND, E / 32 * 9}}) {
        uint64_t bad = 0;
        double maximum = 0;
        for (uint32_t row = 0; row < n; row++)
          for (uint32_t i = 0; i < count; i++) {
            uint32_t j = row * STRIDE + start + i;
            bad += std::bit_cast<uint32_t>(ref[j]) !=
                   std::bit_cast<uint32_t>(got[j]);
            maximum = std::max(maximum, double(std::abs(ref[j] - got[j])));
          }
        std::cout << "T=" << n << " " << name << " differing=" << bad
                  << " max=" << maximum << "\n";
        mismatches += bad;
      }
      vk_check(vkResetFences(ctx.device(), 1, &fence), "reset fence");
      vk_check(vkResetCommandPool(ctx.device(), pool, 0), "reset commands");
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
