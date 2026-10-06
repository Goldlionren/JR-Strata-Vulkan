// Compare only V11 batch register specializations using real weights and the
// frozen plan.
#include "strata/artifact/gguf_reader.hpp"
#include "v10_support.hpp"
#include <bit>
#include <filesystem>
using namespace jr::v10;
const char *shader(uint32_t type) {
  switch (type) {
#define CASE(t)                                                                \
  case t:                                                                      \
    return JR_TEST_T##t;
    CASE(16)
    CASE(17) CASE(18) CASE(20) CASE(21) CASE(22) CASE(42)
#undef CASE
        default : throw std::runtime_error("unsupported expert type");
  }
}
int main(int argc, char **argv) {
  try {
    if (argc < 3 || argc > 4)
      throw std::runtime_error(
          "usage: expert-check native.gguf captured-base.f32 [layer]");
    auto sel = jr::vk::selector_from_env();
    if (sel.vendor_id != 0x8086 || sel.device_id != 0xe211)
      throw std::runtime_error("only JR_VK_DEVICE=8086:e211");
    uint32_t layer = argc == 4 ? std::stoul(argv[3]) : 0;
    constexpr uint32_t E = 2560, FF = 640, K = 10, N = 2;
    auto model = strata::GgufModel::open(argv[1]);
    auto plan = read_cache_plan("data/v10-cache-plan.bin");
    if (layer >= 48)
      throw std::runtime_error("layer out of range");
    const char *names[] = {"ffn_gate_exps.weight", "ffn_up_exps.weight",
                           "ffn_down_exps.weight"};
    std::array<const strata::TensorInfo *, 3> ts;
    std::array<size_t, 3> sh{};
    for (uint32_t r = 0; r < 3; r++) {
      ts[r] =
          model.find("blk." + std::to_string(layer) + "." + names[r], &sh[r]);
      if (!ts[r])
        throw std::runtime_error("missing expert tensor");
    }
    if (ts[0]->type != ts[1]->type)
      throw std::runtime_error("gate/up type mismatch");
    uint32_t gr = row_bytes_for(ts[0]->type, E),
             dr = row_bytes_for(ts[2]->type, FF);
    size_t bytes[] = {FF * gr, FF * gr, E * dr};
    std::vector<uint8_t> ram, vram;
    std::vector<uint32_t> meta(20 * 8);
    for (uint32_t e = 0; e < 20; e++) {
      bool resident = plan.slot[layer * 512 + e] >= 0;
      auto &a = resident ? vram : ram;
      auto *m = meta.data() + e * 8;
      m[0] = resident;
      m[1] = ts[0]->type;
      m[2] = ts[2]->type;
      m[6] = gr;
      m[7] = dr;
      for (uint32_t r = 0; r < 3; r++) {
        size_t at = align_up(a.size(), 4);
        m[3 + r] = at;
        a.resize(at + bytes[r]);
        std::memcpy(a.data() + at,
                    model.shard(sh[r]).tensor_data(*ts[r]) + e * bytes[r],
                    bytes[r]);
      }
    }
    if (ram.empty())
      ram.resize(4);
    if (vram.empty())
      vram.resize(4);
    auto raw = read_bytes(argv[2]);
    auto draft_path =
        std::filesystem::path(argv[2]).parent_path() / "draft.f32";
    auto draft = read_bytes(draft_path.string());
    uint32_t round = raw.size() / 4 - 10240 - 12288;
    if (raw.size() != draft.size() || raw.size() < (round + 720) * 4)
      throw std::runtime_error("captured scratch layout mismatch");
    std::vector<uint8_t> x(N * 720 * 4), h(N * K * 180 * 4);
    for (uint32_t t = 0; t < N; t++) {
      auto &src = t ? draft : raw;
      std::memcpy(x.data() + t * 720 * 4, src.data() + round * 4, 720 * 4);
      for (uint32_t r = 0; r < K; r++)
        std::memcpy(h.data() + (t * K + r) * 180 * 4,
                    src.data() + (round + (r % 4) * 180) * 4, 180 * 4);
    }
    jr::vk::Context ctx(sel);
    auto tables = upload_tables(ctx);
    auto rw = make_imported_host(ctx, ram),
         vw = upload_device(ctx, vram.data(), vram.size()),
         mb = upload_device(ctx, meta.data(), meta.size() * 4),
         xb = upload_device(ctx, x.data(), x.size()),
         hb = upload_device(ctx, h.data(), h.size());
    auto make = [&](uint64_t bytes, bool host = false) {
      return make_buffer(ctx, bytes,
                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                             VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                             VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                         host ? VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
                              : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                         host);
    };
    auto slots = make(20 * 5 * 4, true), output = make(N * K * E * 4),
         readback = make(2 * output.size, true), dummy = make(4);
    std::array<std::array<Pipeline, 2>, 2> pipes;
    uint32_t columns[] = {5, 2};
    for (uint32_t role = 0; role < 2; role++)
      for (uint32_t v = 0; v < 2; v++)
        pipes[role][v] = make_pipeline(
            ctx, shader(ts[role ? 2 : 0]->type),
            {info(rw), info(vw), info(role ? hb : xb), info(output), info(mb),
             info(tables.iq2xxs), info(tables.iq2xs), info(tables.iq2s),
             info(tables.iq3xxs), info(tables.iq3s), info(tables.signs),
             info(tables.iq4), info(slots), info(dummy)},
            32, 32, &columns[v]);
    VkPhysicalDeviceProperties properties;
    vkGetPhysicalDeviceProperties(ctx.physical_device(), &properties);
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
    uint64_t mismatches = 0;
    std::cout << "layer=" << layer << " gate_type=" << ts[0]->type
              << " down_type=" << ts[2]->type
              << " frozen_plan_ram_bytes=" << ram.size()
              << " vram_bytes=" << vram.size() << "\n";
    for (uint32_t overlap : {10u, 5u, 0u})
      for (uint32_t role = 0; role < 2; role++) {
        uint32_t groups = 20 - overlap;
        auto *slot = static_cast<uint32_t *>(slots.mapped);
        std::fill(slot, slot + 100, 0xffffffffu);
        for (uint32_t e = 0; e < groups; e++)
          for (uint32_t t = 0; t < N; t++) {
            uint32_t begin = t * (10 - overlap);
            if (e >= begin && e < begin + 10)
              slot[e * 5 + t] = t * 10 + e - begin;
          }
        uint32_t rows = role ? E : 2 * FF, width = K * rows;
        std::array<uint32_t, 8> p{N * K, E, FF, role, N, role ? K * 180 : 720,
                                  width, 0};
        VkCommandBufferBeginInfo bi{
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        vk_check(vkBeginCommandBuffer(cmd, &bi), "begin");
        vkCmdResetQueryPool(cmd, queries, 0, 4);
        for (uint32_t v = 0; v < 2; v++) {
          auto &pipe = pipes[role][v];
          vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.pipeline);
          vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                  pipe.layout, 0, 1, &pipe.set, 0, nullptr);
          vkCmdPushConstants(cmd, pipe.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                             32, p.data());
          vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                              queries, v * 2);
          for (uint32_t repeat = 0; repeat < 40; repeat++) {
            vkCmdDispatch(cmd, (rows + 3) / 4, groups, 1);
            barrier();
          }
          vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                              queries, v * 2 + 1);
          VkBufferCopy cp{0, v * output.size, output.size};
          vkCmdCopyBuffer(cmd, output.buffer, readback.buffer, 1, &cp);
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
        auto *a = static_cast<float *>(readback.mapped),
             *b = a + output.size / 4;
        uint64_t bad = 0;
        for (uint32_t i = 0; i < N * width; i++)
          bad += std::bit_cast<uint32_t>(a[i]) != std::bit_cast<uint32_t>(b[i]);
        mismatches += bad;
        double scale = properties.limits.timestampPeriod * 1e-6 / 40;
        std::cout << "overlap=" << overlap << " role=" << role
                  << " differing=" << bad
                  << " columns5_ms=" << (stamps[1] - stamps[0]) * scale
                  << " columns2_ms=" << (stamps[3] - stamps[2]) * scale << "\n";
        vk_check(vkResetFences(ctx.device(), 1, &fence), "reset fence");
        vk_check(vkResetCommandPool(ctx.device(), pool, 0), "reset pool");
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
