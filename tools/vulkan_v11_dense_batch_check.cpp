// Real dense weights: preserve RC output bits and time whole-window
// projections.
#include "strata/artifact/gguf_reader.hpp"
#include "v10_dense_encoding.hpp"
#include "v10_support.hpp"
#include <bit>
#include <iostream>
using namespace jr::v10;
struct Push {
  uint32_t op = 0, a = 0, b = 0, c = 0, d = 0, e = 0, f = 0, g = 0, h = 0,
           i = 0, j = 0, k = 0, l = 0, m = 0, n = 0, o = 0;
};
int main(int argc, char **argv) {
  try {
    if (argc < 3 || argc > 6)
      throw std::runtime_error(
          "usage: dense-batch-check native.gguf captured-base.f32 [candidate "
          "SPV] [rows/group] [subgroup]");
    auto sel = jr::vk::selector_from_env();
    if (sel.vendor_id != 0x8086 || sel.device_id != 0xe211)
      throw std::runtime_error("only JR_VK_DEVICE=8086:e211");
    jr::vk::Context ctx(sel);
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(ctx.physical_device(), &props);
    const char *candidate = argc >= 4 ? argv[3] : JR_DENSE_CANDIDATE;
    uint32_t rows_group = argc >= 5 ? std::stoul(argv[4]) : 16,
             sg = argc >= 6 ? std::stoul(argv[5]) : 16;
    auto model = strata::GgufModel::open(argv[1]);
    auto raw = read_bytes(argv[2]);
    auto tables = upload_tables(ctx);
    constexpr uint32_t IN = 0, OUT = 12288, ROUND = OUT + 248320,
                       STRIDE = ROUND + 12288;
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
    auto dummy = make(65536, true), input = make(5ull * STRIDE * 4, true),
         readback = make(10ull * STRIDE * 4, true);
    std::memset(dummy.mapped, 0, dummy.size);
    std::memset(input.mapped, 0, input.size);
    if (raw.size() < 12288 * 4)
      throw std::runtime_error("capture too short");
    for (uint32_t t = 0; t < 5; t++)
      for (uint32_t i = 0; i < 12288; i++)
        static_cast<float *>(input.mapped)[t * STRIDE + i] =
            reinterpret_cast<const float *>(raw.data())[i] * float(1 + .01 * t);
    std::array<Buffer, 2> scratch{make(input.size), make(input.size)};
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
    auto run = [&](Pipeline &p, Push push, uint32_t x, uint32_t y = 1) {
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p.pipeline);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p.layout, 0,
                              1, &p.set, 0, nullptr);
      vkCmdPushConstants(cmd, p.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                         sizeof(push), &push);
      vkCmdDispatch(cmd, x, y, 1);
      barrier();
    };
    for (const char *name :
         {"blk.0.attn_qkv.weight", "blk.0.attn_gate.weight",
          "blk.0.ssm_out.weight", "blk.0.ffn_gate_shexp.weight",
          "blk.0.ffn_up_shexp.weight", "blk.0.ffn_down_shexp.weight",
          "blk.3.attn_q.weight", "blk.3.attn_output.weight", "output.weight"}) {
      size_t shard;
      auto t = model.find(name, &shard);
      if (!t || t->shape.size() != 2)
        throw std::runtime_error(std::string("missing matrix ") + name);
      uint32_t cols = t->shape[0], rows = t->shape[1], type = t->type;
      std::vector<uint8_t> encoded;
      const auto *src = model.shard(shard).tensor_data(*t);
      uint64_t bytes = strata::tensor_payload_bytes(*t);
      if (type == 14 || type == 21 || type == 23) {
        encoded = encode_dense(src, type, t->elements());
        bytes = encoded.size();
        src = encoded.data();
        type = type == 14 ? 100 : type == 21 ? 101 : 102;
      }
      auto weights = upload_device(ctx, src, bytes);
      std::array<Pipeline, 2> qround;
      std::array<std::array<Pipeline, 6>, 2> dense;
      for (uint32_t v = 0; v < 2; v++) {
        std::vector<VkDescriptorBufferInfo> d(12, info(dummy));
        d[0] = info(weights);
        d[1] = info(scratch[v]);
        d[6] = info(tables.iq3s);
        qround[v] = make_pipeline(ctx, JR_DENSE_QROUND, d, sizeof(Push));
        for (uint32_t n = 1; n <= 5; n++) {
          uint32_t format = type | (n << 8);
          dense[v][n] = make_pipeline(ctx, v ? candidate : JR_DENSE_REFERENCE,
                                      d, sizeof(Push), v ? sg : 32, &format);
        }
      }
      for (uint32_t n = 1; n <= 5; n++) {
        VkCommandBufferBeginInfo bi{
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        vk_check(vkBeginCommandBuffer(cmd, &bi), "begin");
        vkCmdResetQueryPool(cmd, queries, 0, 4);
        for (uint32_t v = 0; v < 2; v++) {
          VkBufferCopy cp{0, 0, input.size};
          vkCmdCopyBuffer(cmd, input.buffer, scratch[v].buffer, 1, &cp);
          barrier();
          uint32_t in = IN;
          if (type != 0 && type != 1 && type != 30) {
            run(qround[v],
                {0, IN, ROUND, cols, 0, 0, 0, 0, 0, 0, 0, 16, STRIDE, 0, 0, n},
                cols / 32, n);
            in = ROUND;
          }
          Push p{0, 0, in, OUT, cols,   rows, type, uint32_t(bytes / rows),
                 0, 0, 0,  16,  STRIDE, 0,    0,    n};
          run(dense[v][n], p,
              (rows + (v ? rows_group : 4) - 1) / (v ? rows_group : 4));
          vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                              queries, v * 2);
          for (uint32_t r = 0; r < 20; r++)
            run(dense[v][n], p,
                (rows + (v ? rows_group : 4) - 1) / (v ? rows_group : 4));
          vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                              queries, v * 2 + 1);
          barrier();
          cp = {0, v * input.size, input.size};
          vkCmdCopyBuffer(cmd, scratch[v].buffer, readback.buffer, 1, &cp);
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
        auto a = static_cast<uint32_t *>(readback.mapped),
             b = a + input.size / 4;
        uint64_t bad = 0;
        for (uint32_t row = 0; row < n; row++)
          for (uint32_t i = 0; i < rows; i++)
            bad += a[row * STRIDE + OUT + i] != b[row * STRIDE + OUT + i];
        uint64_t ts[4];
        vk_check(vkGetQueryPoolResults(ctx.device(), queries, 0, 4, sizeof(ts),
                                       ts, 8, VK_QUERY_RESULT_64_BIT),
                 "times");
        double scale = props.limits.timestampPeriod * 1e-6 / 20;
        std::cout << name << " type=" << type << " T=" << n
                  << " differing=" << bad
                  << " rc_ms=" << (ts[1] - ts[0]) * scale
                  << " candidate_ms=" << (ts[3] - ts[2]) * scale << std::endl;
        if (bad)
          return 1;
        vk_check(vkResetFences(ctx.device(), 1, &fence), "reset fence");
        vk_check(vkResetCommandPool(ctx.device(), pool, 0), "reset pool");
      }
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
