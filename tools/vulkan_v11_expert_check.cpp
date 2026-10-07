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
    CASE(17)
    CASE(18)
    CASE(20)
    CASE(21)
    CASE(22)
    CASE(42)
#undef CASE
  default:
    throw std::runtime_error("unsupported expert type");
  }
}
int main(int argc, char **argv) {
  try {
    if (argc < 3 || argc > 13)
      throw std::runtime_error(
          "usage: expert-check native.gguf captured-base.f32 [layer] "
          "[positions] [RC shader directory] [candidate columns] [candidate "
          "subgroup] [candidate shader directory] [prefetch] [copy shader] "
          "[candidate GU rows/group] [candidate down rows/group]");
    auto sel = jr::vk::selector_from_env();
    if (sel.vendor_id != 0x8086 || sel.device_id != 0xe211)
      throw std::runtime_error("only JR_VK_DEVICE=8086:e211");
    uint32_t layer = argc >= 4 ? std::stoul(argv[3]) : 0;
    constexpr uint32_t E = 2560, FF = 640, K = 10;
    uint32_t N = argc >= 5 ? std::stoul(argv[4]) : 2;
    std::string reference = argc >= 6 ? argv[5] : "";
    uint32_t candidate_columns = argc >= 7 ? std::stoul(argv[6]) : 5;
    uint32_t candidate_subgroup = argc >= 8 ? std::stoul(argv[7]) : 32;
    if (N < 1 || N > 5 ||
        (!reference.empty() &&
         (candidate_columns != 0 &&
          (candidate_columns < N || candidate_columns > 5))))
      throw std::runtime_error("invalid position/column count");
    std::string candidate_directory = argc >= 9 ? argv[8] : "";
    bool current_reference = reference == "current";
    uint32_t candidate_rows = argc >= 12 ? std::stoul(argv[11]) : 4;
    if (candidate_rows != 4 && candidate_rows != 8 && candidate_rows != 16 &&
        candidate_rows != 32)
      throw std::runtime_error("invalid candidate rows/group");
    uint32_t candidate_down_rows =
        argc >= 13 ? std::stoul(argv[12]) : candidate_rows;
    if (candidate_down_rows != 4 && candidate_down_rows != 8 &&
        candidate_down_rows != 16 && candidate_down_rows != 32)
      throw std::runtime_error("invalid candidate down rows/group");
    bool prefetch = argc >= 10 && std::string(argv[9]) == "prefetch";
    if (prefetch && N > 4)
      throw std::runtime_error(
          "prefetch is limited to windows with an unused snapshot");
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
    std::vector<uint32_t> meta(50 * 8);
    for (uint32_t e = 0; e < 50; e++) {
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
    if (!ctx.caps().subgroup_size_control ||
        (candidate_subgroup != 0 &&
         candidate_subgroup < ctx.caps().min_subgroup_size) ||
        candidate_subgroup > ctx.caps().max_subgroup_size)
      throw std::runtime_error("requested subgroup unsupported");
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
                             VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                             VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
                         host ? VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
                              : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                         host);
    };
    auto slots = make(520 * 4, true), output = make(N * K * E * 4),
         readback = make(2 * output.size, true), dummy = make(4);
    auto staged = make(prefetch ? 1125ull * 1024 * 1024 / 10 : 4),
         staged_meta = make(meta.size() * 4, true);
    auto grouped_meta = make(50 * 8 * 4, true),
         input_routes = make(N * 128 * 4, true);
    auto prefetch_pipeline = make_pipeline(
        ctx,
        argc >= 11 && std::string(argv[10]) != "default" ? argv[10]
                                                         : JR_TEST_PREFETCH,
        {info(rw), info(grouped_meta), info(slots), info(staged),
         info(staged_meta)},
        8);
    auto prefetch_group = make_pipeline(ctx, JR_TEST_PREFETCH_GROUP,
                                        {info(input_routes), info(grouped_meta),
                                         info(slots), info(staged_meta)},
                                        16);
    std::array<std::array<Pipeline, 2>, 2> pipes;
    auto descriptors = [&](uint32_t role, bool stage = false) {
      return std::vector<VkDescriptorBufferInfo>{info(stage ? staged : rw),
                                                 info(vw),
                                                 info(role ? hb : xb),
                                                 info(output),
                                                 info(stage ? staged_meta : mb),
                                                 info(tables.iq2xxs),
                                                 info(tables.iq2xs),
                                                 info(tables.iq2s),
                                                 info(tables.iq3xxs),
                                                 info(tables.iq3s),
                                                 info(tables.signs),
                                                 info(tables.iq4),
                                                 info(slots),
                                                 info(dummy)};
    };
    for (uint32_t role = 0; role < 2; role++) {
      uint32_t type = ts[role ? 2 : 0]->type;
      std::string candidate_path =
          candidate_directory.empty()
              ? shader(type)
              : (std::filesystem::path(candidate_directory) /
                 ("v11_expert_t" + std::to_string(type) + ".comp.spv"))
                    .string();
      if (!candidate_directory.empty() && !role && candidate_rows > 4) {
        auto wide = std::filesystem::path(candidate_directory) /
                    ("v11_verify_expert" + std::to_string(candidate_rows * 8) +
                     "_t" + std::to_string(type) + ".comp.spv");
        if (std::filesystem::exists(wide))
          candidate_path = wide.string();
      }
      std::string ref_path =
          (std::filesystem::path(reference) /
           ("v11_expert_t" + std::to_string(type) + ".comp.spv"))
              .string();
      uint32_t selected_columns =
          candidate_columns
              ? candidate_columns
              : (!role
                     ? (N == 3 ? 3
                               : (N == 2 && (type == 18 || type == 21) ? 2 : 5))
                     : 5);
      uint32_t selected_subgroup =
          candidate_subgroup
              ? candidate_subgroup
              : (role ? 32
                      : ((N == 3 && (type == 16 || type == 17 || type == 22)) ||
                                 (N == 2 && (type == 18 || type == 21))
                             ? 32
                             : 16));
      for (uint32_t v = 0; v < 2; v++) {
        uint32_t columns =
            v || current_reference
                ? selected_columns
                : (!role && N == 2 && (type == 18 || type == 21) ? 2 : 5);
        if (v || current_reference)
          columns |= N << 8;
        pipes[role][v] = make_pipeline(
            ctx,
            !v ? (current_reference ? shader(type) : ref_path.c_str())
               : candidate_path.c_str(),
            descriptors(role, prefetch && (v || current_reference)), 32,
            v || current_reference ? selected_subgroup : 32, &columns);
      }
    }
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
    qi.queryCount = 6;
    VkQueryPool queries;
    vk_check(vkCreateQueryPool(ctx.device(), &qi, nullptr, &queries),
             "queries");
    auto barrier = [&]() {
      VkMemoryBarrier b{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
      b.srcAccessMask =
          VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
      b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                        VK_ACCESS_TRANSFER_READ_BIT |
                        VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
      vkCmdPipelineBarrier(cmd,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                               VK_PIPELINE_STAGE_TRANSFER_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                               VK_PIPELINE_STAGE_TRANSFER_BIT |
                               VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
                           0, 1, &b, 0, nullptr, 0, nullptr);
    };
    uint64_t mismatches = 0;
    std::cout << "layer=" << layer << " gate_type=" << ts[0]->type
              << " down_type=" << ts[2]->type
              << " subgroup=" << candidate_subgroup << " positions=" << N
              << " prefetch=" << prefetch << "\n";
    auto bind = [&](Pipeline &pipe, const std::array<uint32_t, 8> &push) {
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.pipeline);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.layout,
                              0, 1, &pipe.set, 0, nullptr);
      vkCmdPushConstants(cmd, pipe.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 32,
                         push.data());
    };
    for (uint32_t overlap : {10u, 5u, 0u})
      for (uint32_t role = 0; role < 2; role++) {
        uint32_t groups = 10 + (N - 1) * (10 - overlap),
                 rows = role ? E : 2 * FF, width = K * rows;
        auto slot = static_cast<uint32_t *>(slots.mapped);
        std::fill(slot, slot + 520, 0xffffffffu);
        for (uint32_t e = 0; e < groups; e++)
          for (uint32_t t = 0; t < N; t++) {
            uint32_t begin = t * (10 - overlap);
            if (e >= begin && e < begin + 10)
              slot[e * 5 + t] = t * 10 + e - begin;
          }
        uint32_t host_count = 0;
        for (uint32_t rank = 0; rank < groups; rank++)
          if (meta[rank * 8] == 0)
            slot[250 + host_count++] = rank;
        std::array<uint32_t, 520> native_expected;
        std::copy(slot, slot + 520, native_expected.begin());
        native_expected[508] = 32;
        native_expected[509] = host_count;
        native_expected[510] = 3;
        native_expected[512] = 2 * FF / 4;
        native_expected[513] = groups;
        native_expected[514] = 1;
        native_expected[516] = E / 4;
        native_expected[517] = groups;
        native_expected[518] = 1;
        for (uint32_t t = 0; t < N; t++)
          for (uint32_t k = 0; k < K; k++)
            std::memcpy(static_cast<uint32_t *>(input_routes.mapped) + t * 128 +
                            k * 8,
                        meta.data() + (t * (10 - overlap) + k) * 8, 32);
        std::array<uint32_t, 8> push{
            N * K, E, FF, role, N, role ? K * 180 : 720, width, 0};
        VkCommandBufferBeginInfo bi{
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        vk_check(vkBeginCommandBuffer(cmd, &bi), "begin");
        vkCmdResetQueryPool(cmd, queries, 0, 6);
        if (prefetch) {
          std::array<uint32_t, 4> gp{N, 128, 2 * FF / 4, E / 4};
          vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            prefetch_group.pipeline);
          vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                  prefetch_group.layout, 0, 1,
                                  &prefetch_group.set, 0, nullptr);
          vkCmdPushConstants(cmd, prefetch_group.layout,
                             VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, gp.data());
          vkCmdDispatch(cmd, 1, 1, 1);
          barrier();
        }
        for (uint32_t v = 0; v < 2; v++) {
          vkCmdFillBuffer(cmd, output.buffer, 0, output.size, 0x7fc00001u);
          barrier();
          if (prefetch && (v || current_reference)) {
            std::array<uint32_t, 2> cp{uint32_t(bytes[0]), uint32_t(bytes[2])};
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                              prefetch_pipeline.pipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    prefetch_pipeline.layout, 0, 1,
                                    &prefetch_pipeline.set, 0, nullptr);
            vkCmdPushConstants(cmd, prefetch_pipeline.layout,
                               VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, cp.data());
            vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                queries, 2);
            for (uint32_t repeat = 0; repeat < 40; repeat++) {
              vkCmdDispatchIndirect(cmd, slots.buffer, 508 * 4);
              barrier();
            }
            vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                queries, 3);
          }
          bind(pipes[role][v], push);
          vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                              queries, v * (prefetch ? 4 : 2));
          for (uint32_t repeat = 0; repeat < 40; repeat++) {
            vkCmdDispatch(
                cmd,
                (rows +
                 (v ? (role ? candidate_down_rows : candidate_rows) : 4) - 1) /
                    (v ? (role ? candidate_down_rows : candidate_rows) : 4),
                groups, 1);
            barrier();
          }
          vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                              queries, v * (prefetch ? 4 : 2) + 1);
          VkBufferCopy cp{0, v * output.size, output.size};
          vkCmdCopyBuffer(cmd, output.buffer, readback.buffer, 1, &cp);
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
        vk_check(
            vkWaitForFences(ctx.device(), 1, &fence, VK_TRUE, 120000000000ull),
            "wait");
        if (prefetch) {
          auto actual = static_cast<uint32_t *>(slots.mapped);
          if (std::memcmp(native_expected.data(), actual,
                          native_expected.size() * 4) != 0)
            throw std::runtime_error("GPU native slots/host list/control "
                                     "differs from independent CPU oracle");
          if (actual[508] != 32 || actual[509] != host_count ||
              actual[510] != 3)
            throw std::runtime_error(
                "native prefetch indirect record mismatch");
          if (std::memcmp(grouped_meta.mapped, meta.data(), groups * 32) != 0)
            throw std::runtime_error("native GPU grouping metadata mismatch");
          auto staged_words = static_cast<uint32_t *>(staged_meta.mapped);
          for (uint32_t rank = 0; rank < groups; rank++)
            for (uint32_t f = 0; f < 8; f++) {
              uint32_t expected = meta[rank * 8 + f];
              if (meta[rank * 8] == 0 && f >= 3 && f <= 5)
                expected =
                    rank * (2 * bytes[0] + bytes[2]) + (f - 3) * bytes[0];
              if (staged_words[rank * 8 + f] != expected)
                throw std::runtime_error("GPU staging metadata mismatch");
            }
        }
        uint64_t stamps[6];
        vk_check(vkGetQueryPoolResults(ctx.device(), queries, 0,
                                       prefetch ? 6 : 4, sizeof(stamps), stamps,
                                       8, VK_QUERY_RESULT_64_BIT),
                 "timestamps");
        auto aa = static_cast<uint32_t *>(readback.mapped),
             bb = aa + output.size / 4;
        uint64_t bad = 0;
        for (uint32_t i = 0; i < N * width; i++)
          bad += aa[i] != bb[i];
        mismatches += bad;
        double scale = properties.limits.timestampPeriod * 1e-6 / 40;
        std::cout << "overlap=" << overlap << " role=" << role
                  << " differing=" << bad
                  << " reference_ms=" << (stamps[1] - stamps[0]) * scale
                  << " candidate_ms="
                  << (stamps[prefetch ? 5 : 3] - stamps[prefetch ? 4 : 2]) *
                         scale
                  << " prefetch_ms="
                  << (prefetch ? (stamps[3] - stamps[2]) * scale : 0)
                  << std::endl;
        if (bad)
          return 1;
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
