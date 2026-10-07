// Intel Xe2 / Swift minimum token runtime. No CUDA/SYCL execution dependency.
#include "strata/artifact/dequant.hpp"
#include "v10_dense_encoding.hpp"
#include "v10_support.hpp"
#include <array>
#include <bit>
#include <cmath>
#include <deque>
#include <fcntl.h>
#include <filesystem>
#include <iomanip>
#include <map>
#include <memory>
#include <numeric>
#include <sstream>
#include <sys/mman.h>
#include <sys/resource.h>
#include <unistd.h>

namespace jr::v11 {
using namespace jr::v10;
namespace fs = std::filesystem;
constexpr uint32_t E = 2560, HC = 10240, FF = 640, K = 10, VOCAB = 248320;
jr::vk::DeviceSelector b60_selector() {
  auto s = jr::vk::selector_from_env();
  if (s.vendor_id != 0x8086 || s.device_id != 0xe211)
    throw std::runtime_error("V11 only permits JR_VK_DEVICE=8086:e211");
  return s;
}
struct Options {
  std::string pack, native, ple, residency, ids, logits, dump, route_profile;
  std::string mtp = "/data/strata-lab/data/mtp/rt", forced;
  uint32_t spec = 4;
  float min_p = 0.5f;
  int force_reject = -1;
  uint32_t context = 8192, max_new = 1, stats_every = 128;
  bool worker = false, checks = false, state_checks = false, ignore_eos = false,
       profile = false, draft_check = false, adaptive_spec = true;
};
Options options(int argc, char **argv) {
  Options a;
  for (int i = 1; i < argc; i++) {
    std::string k = argv[i];
    auto need = [&]() {
      if (i + 1 == argc)
        throw std::runtime_error("missing argument: " + k);
      return std::string(argv[++i]);
    };
    if (k == "--mtp")
      a.mtp = need();
    else if (k == "--draft-check")
      a.draft_check = true;
    else if (k == "--fixed-spec")
      a.adaptive_spec = false;
    else if (k == "--spec")
      a.spec = std::stoul(need());
    else if (k == "--spec-min-p")
      a.min_p = std::stof(need());
    else if (k == "--force-drafts")
      a.forced = need();
    else if (k == "--force-reject")
      a.force_reject = std::stoi(need());
    else if (k == "--pack")
      a.pack = need();
    else if (k == "--native")
      a.native = need();
    else if (k == "--ple-gguf")
      a.ple = need();
    else if (k == "--residency")
      a.residency = need();
    else if (k == "--max-context")
      a.context = std::stoul(need());
    else if (k == "--max-new")
      a.max_new = std::stoul(need());
    else if (k == "--ids")
      a.ids = need();
    else if (k == "--logits-out")
      a.logits = need();
    else if (k == "--dump-dir")
      a.dump = need();
    else if (k == "--stats-every")
      a.stats_every = std::stoul(need());
    else if (k == "--kv") {
      if (need() != "fp16")
        throw std::runtime_error("V11 requires --kv fp16");
    } else if (k == "--greedy") {
    } else if (k == "--worker")
      a.worker = true;
    else if (k == "--checks")
      a.checks = true;
    else if (k == "--state-checks") {
      a.state_checks = true;
      a.checks = true;
    } else if (k == "--ignore-eos")
      a.ignore_eos = true;
    else if (k == "--profile")
      a.profile = true;
    else if (k == "--route-profile")
      a.route_profile = need();
    else
      throw std::runtime_error("unknown runtime option: " + k);
  }
  if (a.spec > 4 || !std::isfinite(a.min_p) || a.min_p < 0 || a.min_p > 1 ||
      a.force_reject < -1 || a.force_reject > 3)
    throw std::runtime_error("invalid MTP spec/probability/rejection option");
  if (a.pack.empty() || a.native.empty() || a.residency.empty())
    throw std::runtime_error("--pack, --native and --residency are required");
  if (a.ple.empty())
    a.ple = a.native;
  if (a.context < 1 || a.context > 8192)
    throw std::runtime_error("V11 context must be 1..8192");
  if (!a.max_new || !a.stats_every)
    throw std::runtime_error("max-new and stats-every must be positive");
  a.checks = a.checks || std::getenv("JR_VK_CHECKS");
  return a;
}

struct Tensor {
  uint32_t off = 0, type = 0, cols = 0, rows = 0, row_bytes = 0;
};
struct Push {
  uint32_t op = 0, a = 0, b = 0, c = 0, d = 0, e = 0, f = 0, g = 0, h = 0,
           i = 0, j = 0, k = 0, l = 0, m = 0, n = 0, o = 0;
};
struct Work {
  uint32_t end = 0;
  uint32_t take(uint32_t n) {
    uint32_t at = end;
    end += (n + 63) & ~63u;
    return at;
  }
  uint32_t R = take(HC), xn = take(HC), lo = take(320), gate = take(HC),
           inject = take(4), mixed = take(E), block = take(E), qkv = take(HC),
           h = take(HC), z = take(6144), alpha = take(48), beta = take(48),
           rec = take(6144), y = take(6144), qfull = take(12288),
           q = take(6144), kcur = take(512), vcur = take(512),
           idxraw = take(128), idxq = take(512), scores = take(2050),
           selected = take(2051), partial = take(24 * 65 * 258),
           attn = take(6144), shg = take(640), shu = take(640), shh = take(640),
           sho = take(E), shgate = take(1), moe = take(E), head = take(VOCAB),
           pleemb = take(E), plekey = take(HC), plequery = take(HC),
           pleval = take(E), plegated = take(HC), plenorm = take(HC),
           round = take(12288), Rin = take(HC);
};
struct RowPath {
  Pipeline ops, gdn, attn, gu, down, qround, router, hc, selection;
  std::map<uint32_t, Pipeline> dense_gemv, dense_small;
};
struct Layer {
  Buffer dense, ram, vram, plan, state, kv;
  std::map<std::string, Tensor> tensors;
  std::array<RowPath, 6> paths;
  std::array<std::map<uint32_t, Pipeline>, 6> batch_dense, batch_small;
  Pipeline batch_gdn;
  std::array<Pipeline, 6> batch_gu, batch_down, batch_hc, batch_hc_down;
  Pipeline batch_norm, batch_qround, batch_gdn_prepare, batch_qsa_ops,
      batch_attn, batch_attn_tile;
  Pipeline batch_router, batch_selection, batch_gdn_output, batch_qsa_prepare,
      batch_qsa_score, verify_prefetch;
  uint32_t gu_type = 0, d_type = 0;
};
struct Checkpoint {
  uint32_t offset;
  int layer;
  std::string stage;
};
struct TokenResult {
  uint32_t token;
  float logit;
  double total_ms;
  std::array<double, 6> gpu_ms{};
};

class Runtime {
  Options a_;
  jr::vk::Context ctx_;
  strata::GgufModel model_;
  CachePlan plan_;
  Tables tables_;
  Work w_;
  Buffer scratch_, ctl_, diag_, route_, weights_, gu_, hq_, parts_, dummy_,
      head_host_, embedding_ram_, dense_grid_;
  std::array<Layer, 50> layers_;
  std::array<Pipeline, 6> swiglu_, combine_, embedding_;
  Buffer snapshots_, draft_vocab_, group_meta_, group_slots_, mtp_partial_;
  Pipeline group_routes_, batch_swiglu_, batch_combine_;
  Buffer draft_debug_;
  VkCommandBuffer draft_debug_cmd_ = VK_NULL_HANDLE;
  bool draft_dumped_ = false;
  uint32_t rows_ = 1, active_row_ = 0, verify_rows_ = 0, draft_vocab_count_ = 0;
  uint64_t ctl_stride_ = 0, route_stride_ = 512, weights_stride_ = 256,
           gu_stride_ = 51200, hq_stride_ = 7424, parts_stride_ = 102400,
           snapshot_stride_ = 0;
  std::array<uint64_t, 48> snapshot_offset_{}, snapshot_bytes_{};
  double peak_vram_ = 0, min_headroom_ = 1e30;
  std::array<VkCommandBuffer, 6> verify_cmd_{}, restore_cmd_{}, commit_cmd_{},
      catchup_cmd_{};
  VkCommandBuffer draft_cmd_ = VK_NULL_HANDLE, seed_cmd_ = VK_NULL_HANDLE;
  VkQueryPool batch_queries_ = VK_NULL_HANDLE;
  uint32_t batch_nq_ = 0;
  std::array<std::vector<std::pair<uint32_t, std::string>>, 6> batch_marks_;
  std::vector<uint32_t> forced_;
  std::array<uint32_t, 4> test_override_{};
  uint32_t test_override_count_ = 0;
  struct Metrics {
    uint64_t proposed = 0, accepted = 0, windows = 0, rollbacks = 0,
             committed = 0, proposal_hash = 14695981039346656037ull;
    std::array<uint64_t, 4> depth_proposed{}, depth_accepted{};
    std::array<uint64_t, 5> window_sizes{};
    std::array<double, 5> window_cost{}, window_commits{};
    uint32_t adaptive_max_drafts = 4;
    double draft_ms = 0, verify_ms = 0, rollback_ms = 0, catchup_ms = 0,
           total_ms = 0;
  } metrics_;
  RowPath &path(uint32_t l) { return layers_[l].paths[active_row_]; }
  VkDescriptorBufferInfo slice(Buffer &b, uint64_t stride, uint32_t row) {
    auto d = info(b, stride);
    d.offset = stride * row;
    return d;
  }
  VkCommandPool pool_ = VK_NULL_HANDLE;
  VkCommandBuffer cmd_ = VK_NULL_HANDLE, fast_cmd_ = VK_NULL_HANDLE;
  bool timing_enabled_ = true;
  VkFence fence_ = VK_NULL_HANDLE;
  VkQueryPool queries_ = VK_NULL_HANDLE;
  std::vector<std::array<uint32_t, 3>> timings_;
  std::vector<std::pair<uint32_t, std::string>> kernel_timings_;
  std::string active_stage_;
  uint64_t route_hits_ = 0, route_count_ = 0;
  std::array<uint64_t, 48 * 512> route_counts_{};
  std::vector<Checkpoint> checkpoints_;
  std::vector<uint64_t> timestamps_;
  double timestamp_period_ = 1;
  uint32_t nq_ = 0, position_ = 0;
  uint32_t rounded_input_ = UINT32_MAX, rounded_cols_ = 0;
  uint32_t batch_input_ = UINT32_MAX, batch_cols_ = 0;
  std::array<uint32_t, 16> previous_{};
  uint32_t previous_count_ = 0;
  int plefd_ = -1;
  uint64_t pleoff_ = 0;
  const strata::TensorInfo *embed_ = nullptr;
  size_t embed_shard_ = 0;
  std::array<uint64_t, 3> multipliers_{};
  std::array<uint64_t, 16> vocab_{}, offsets_{};
  std::vector<std::array<float, 160>> plecache_{65536};
  std::vector<uint32_t> plecache_ids_ =
      std::vector<uint32_t>(65536, UINT32_MAX);
  static constexpr size_t ple_cache_rows = 65536;
  Buffer buffer(uint64_t n, bool mapped = false, bool indirect = false) {
    return make_buffer(ctx_, std::max<uint64_t>(n, 4),
                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                           VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                           VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                           (indirect ? VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT : 0),
                       mapped ? VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
                              : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                       mapped);
  }
  const strata::TensorInfo &tensor(const std::string &name, size_t &shard) {
    auto t = model_.find(name, &shard);
    if (!t || !model_.in_bounds(*t, shard))
      throw std::runtime_error("missing or invalid GGUF tensor " + name);
    return *t;
  }
  void verify_geometry() {
    auto &md = model_.meta();
    auto require = [&](const char *key, uint64_t value) {
      auto p = md.get(key);
      if (!p || p->u != value)
        throw std::runtime_error(std::string("unsupported geometry: ") + key);
    };
    auto arch = md.get("general.architecture");
    if (!arch || arch->s != "qwen4exp")
      throw std::runtime_error("V11 requires qwen4exp");
    require("qwen4exp.block_count", 48);
    require("qwen4exp.embedding_length", E);
    require("qwen4exp.expert_count", 512);
    require("qwen4exp.expert_used_count", 10);
    require("qwen4exp.attention.head_count", 24);
    require("qwen4exp.attention.head_count_kv", 2);
    require("qwen4exp.hyper_connection.count", 4);
    require("qwen4exp.hyper_connection.low_rank", 320);
    require("qwen4exp.full_attention_interval", 4);
    require("qwen4exp.ple.eos_token_id", 248044);
    auto read_array = [&](const char *key, auto &dest) {
      auto p = md.get(key);
      if (!p || p->items.size() != dest.size())
        throw std::runtime_error(std::string("invalid metadata ") + key);
      for (size_t i = 0; i < dest.size(); i++)
        dest[i] = p->items[i].u;
    };
    read_array("qwen4exp.ple.layer_multipliers", multipliers_);
    read_array("qwen4exp.ple.head_vocab_sizes", vocab_);
    read_array("qwen4exp.ple.head_offsets", offsets_);
    if (plan_.n_layers != 48 || plan_.n_expert != 512 ||
        plan_.slot.size() != 48 * 512)
      throw std::runtime_error("residency dimensions disagree with model");
    std::vector<int> seen(plan_.n_resident, 0);
    for (auto x : plan_.slot) {
      if (x < -1 || x >= int(seen.size()))
        throw std::runtime_error("invalid residency slot");
      if (x >= 0 && seen[x]++)
        throw std::runtime_error("duplicate residency slot");
    }
    if (std::accumulate(seen.begin(), seen.end(), 0) != int(plan_.n_resident))
      throw std::runtime_error("residency count mismatch");
  }
  void memory_report(const std::string &where) {
    VkPhysicalDeviceMemoryBudgetPropertiesEXT budget{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
    VkPhysicalDeviceMemoryProperties2 properties{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2};
    properties.pNext = &budget;
    vkGetPhysicalDeviceMemoryProperties2(ctx_.physical_device(), &properties);
    for (uint32_t i = 0; i < properties.memoryProperties.memoryHeapCount; i++)
      if (properties.memoryProperties.memoryHeaps[i].flags &
          VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
        peak_vram_ =
            std::max(peak_vram_, double(budget.heapUsage[i]) / (1ull << 30));
        min_headroom_ = std::min(min_headroom_, (double(budget.heapBudget[i]) -
                                                 double(budget.heapUsage[i])) /
                                                    (1ull << 30));
        if (where == "MTP ready" && min_headroom_ < 0.5)
          throw std::runtime_error(
              "MTP requires at least 512 MiB stable VRAM headroom");
        std::cerr << "VRAM " << where << ": "
                  << double(budget.heapUsage[i]) / (1ull << 30)
                  << " GiB, budget "
                  << double(budget.heapBudget[i]) / (1ull << 30)
                  << " GiB, headroom "
                  << (double(budget.heapBudget[i]) -
                      double(budget.heapUsage[i])) /
                         (1ull << 30)
                  << " GiB\n";
      }
    std::ifstream statm("/proc/self/statm");
    uint64_t pages = 0, resident = 0;
    statm >> pages >> resident;
    struct rusage usage{};
    getrusage(RUSAGE_SELF, &usage);
    std::cerr << "Host RSS " << where << ": "
              << double(resident) * sysconf(_SC_PAGESIZE) / (1ull << 30)
              << " GiB, peak " << double(usage.ru_maxrss) * 1024 / (1ull << 30)
              << " GiB\n";
  }
  void release_source_pages(const strata::GgufFile &file,
                            const strata::TensorInfo &t) {
    // The copied payload is now owned by device/imported buffers. Discard only
    // this process's file mapping pages, so loading cannot retain a second
    // resident copy of the whole expert set alongside imported host memory.
    const auto begin = reinterpret_cast<uintptr_t>(file.tensor_data(t));
    const auto page = static_cast<uintptr_t>(sysconf(_SC_PAGESIZE));
    const auto start = begin & ~(page - 1);
    const auto finish = align_up(begin + strata::tensor_payload_bytes(t), page);
    if (madvise(reinterpret_cast<void *>(start), finish - start,
                MADV_DONTNEED) != 0)
      throw std::runtime_error("cannot release copied GGUF mapping pages");
  }
  void load_dense(uint32_t l) {
    std::vector<uint8_t> data;
    auto &layer = layers_[l];
    for (size_t sh = 0; sh < model_.size(); sh++)
      for (auto &t : model_.shard(sh).tensors()) {
        bool select = l < 48
                          ? t.name.starts_with("blk." + std::to_string(l) + ".")
                          : t.name.starts_with("output");
        if (!select || t.name.find("_exps.") != std::string::npos)
          continue;
        if (!model_.in_bounds(t, sh))
          throw std::runtime_error("invalid dense payload " + t.name);
        if (t.type != 0 && t.type != 1 && t.type != 12 && t.type != 13 &&
            t.type != 14 && t.type != 20 && t.type != 21 && t.type != 23 &&
            t.type != 30 && t.type != 42)
          throw std::runtime_error("unsupported V11 dense type " + t.name +
                                   " " + t.type_name());
        uint64_t bytes = strata::tensor_payload_bytes(t);
        uint32_t storage_type = t.type;
        std::vector<uint16_t> bf16;
        std::vector<uint8_t> encoded;
        if (t.type == 0 && t.name.ends_with("ffn_gate_inp.weight")) {
          bf16.resize(t.elements());
          const auto *src = model_.shard(sh).tensor_data(t);
          for (size_t i = 0; i < bf16.size(); i++) {
            uint32_t bits;
            std::memcpy(&bits, src + i * 4, 4);
            if (bits & 65535)
              throw std::runtime_error(
                  "router F32 values are not losslessly BF16");
            bf16[i] = bits >> 16;
          }
          bytes = bf16.size() * 2;
          storage_type = 30;
        }
        if ((t.type == 14 || t.type == 23 || t.type == 21) &&
            t.name != "token_embd.weight") {
          encoded = encode_dense(model_.shard(sh).tensor_data(t), t.type,
                                 t.elements());
          bytes = encoded.size();
          storage_type = t.type == 14 ? 100 : t.type == 21 ? 101 : 102;
        }
        size_t at = align_up(data.size(), 64);
        if (at + bytes > UINT32_MAX)
          throw std::runtime_error("dense segment >4GiB");
        data.resize(at + bytes + 4, 0);
        std::memcpy(data.data() + at,
                    !encoded.empty() ? encoded.data()
                    : bf16.empty()
                        ? model_.shard(sh).tensor_data(t)
                        : reinterpret_cast<const uint8_t *>(bf16.data()),
                    bytes);
        int block = 0, sz = 0;
        if (storage_type == 100) {
          block = 256;
          sz = 212;
        } else if (storage_type == 101 || storage_type == 102 ||
                   storage_type == 103) {
          block = 256;
          sz = 140;
        } else if (!strata::block_geometry(storage_type, block, sz))
          throw std::runtime_error("unsupported block geometry");
        uint32_t cols = t.shape[0], rows = t.shape.size() > 1 ? t.shape[1] : 1;
        if (storage_type == 102 && rows <= 1024)
          storage_type = 103;
        if (cols % block)
          throw std::runtime_error("unaligned dense row " + t.name);
        layer.tensors[t.name] = {uint32_t(at), storage_type, cols, rows,
                                 cols / uint32_t(block) * uint32_t(sz)};
      }
    layer.dense = upload_device(ctx_, data.data(), data.size());
    for (auto &[name, unused] : layer.tensors) {
      size_t sh;
      const auto &t = tensor(name, sh);
      release_source_pages(model_.shard(sh), t);
    }
  }
  void load_experts(uint32_t l) {
    auto &layer = layers_[l];
    std::array<const strata::TensorInfo *, 3> ts;
    std::array<size_t, 3> shards{};
    const char *names[] = {"ffn_gate_exps.weight", "ffn_up_exps.weight",
                           "ffn_down_exps.weight"};
    for (int r = 0; r < 3; r++) {
      ts[r] = &tensor("blk." + std::to_string(l) + "." + names[r], shards[r]);
      auto &t = *ts[r];
      std::vector<uint64_t> shape = r < 2 ? std::vector<uint64_t>{E, FF, 512}
                                          : std::vector<uint64_t>{FF, E, 512};
      if (t.shape != shape)
        throw std::runtime_error("expert geometry mismatch " + t.name);
    }
    if (ts[0]->type != ts[1]->type)
      throw std::runtime_error("mixed gate/up types");
    layer.gu_type = ts[0]->type;
    layer.d_type = ts[2]->type;
    size_t gr = row_bytes_for(layer.gu_type, E),
           dr = row_bytes_for(layer.d_type, FF);
    size_t sizes[] = {FF * gr, FF * gr, E * dr};
    std::vector<uint8_t> ram, vram;
    std::vector<uint32_t> meta(512 * 8);
    for (uint32_t e = 0; e < 512; e++) {
      bool resident = plan_.slot[l * 512 + e] >= 0;
      auto &arena = resident ? vram : ram;
      auto *m = meta.data() + e * 8;
      m[0] = resident;
      m[1] = layer.gu_type;
      m[2] = layer.d_type;
      m[6] = gr;
      m[7] = dr;
      for (int r = 0; r < 3; r++) {
        size_t at = align_up(arena.size(), 4);
        m[3 + r] = at;
        arena.resize(at + sizes[r]);
        std::memcpy(arena.data() + at,
                    model_.shard(shards[r]).tensor_data(*ts[r]) + sizes[r] * e,
                    sizes[r]);
      }
    }
    // Each layer is a bounded, independent segment, preserving V8's 32-bit byte
    // addressing.
    if (ram.empty())
      ram.resize(4);
    layer.ram = make_imported_host(ctx_, ram);
    if (vram.empty())
      vram.resize(4);
    layer.vram = upload_device(ctx_, vram.data(), vram.size());
    layer.plan = upload_device(ctx_, meta.data(), meta.size() * 4);
    for (int r = 0; r < 3; r++)
      release_source_pages(model_.shard(shards[r]), *ts[r]);
  }
  void load_ple() {
    auto paths = strata::gguf_split_paths(a_.ple);
    for (auto &path : paths) {
      strata::GgufFile file(path);
      auto t = file.find("per_layer_token_embd.weight");
      if (!t)
        continue;
      if (t->type != 20 || t->shape != std::vector<uint64_t>{160, 320001536} ||
          file.data_start() + t->offset + strata::tensor_payload_bytes(*t) >
              file.file_size())
        throw std::runtime_error("unsupported PLE table");
      pleoff_ = file.data_start() + t->offset;
      plefd_ = open(path.c_str(), O_RDONLY | O_CLOEXEC);
      if (plefd_ < 0)
        throw std::runtime_error("cannot open PLE " + path);
      std::cerr
          << "PLE source: " << path
          << " per_layer_token_embd.weight (IQ4_NL, bounded 65536-row cache)\n";
      return;
    }
    throw std::runtime_error("PLE tensor not found");
  }
  std::vector<VkDescriptorBufferInfo> desc(uint32_t l, uint32_t row = 0) {
    auto &z = layers_[l];
    return {info(z.dense),
            slice(scratch_, w_.end * 4, row),
            slice(ctl_, ctl_stride_, row),
            info(z.state),
            info(z.kv),
            slice(diag_, 65536, row),
            info(dense_grid_),
            info(z.plan),
            slice(route_, route_stride_, row),
            slice(weights_, weights_stride_, row),
            info(snapshots_),
            info(draft_vocab_)};
  }
  void pipelines(uint32_t l, uint32_t row = 0) {
    auto &z = layers_[l];
    auto d = desc(l, row);
    auto &q = z.paths[row];
    q.ops = make_pipeline(ctx_, JR_VK_V10_OPS_SHADER, d, sizeof(Push));
    q.qround = make_pipeline(ctx_, JR_VK_V10_QROUND_SHADER, d, sizeof(Push));
    q.hc = make_pipeline(ctx_, JR_VK_V10_HC_SHADER, d, sizeof(Push));
    if (l != 48)
      q.router = make_pipeline(ctx_, JR_VK_V10_ROUTER_SHADER, d, sizeof(Push));
    for (auto &[name, t] : z.tensors) {
      if ((t.type == 0 || t.type == 30 || t.type == 1) &&
          !q.dense_small.contains(t.type))
        q.dense_small.emplace(t.type,
                              make_pipeline(ctx_, JR_VK_V10_SMALL_SHADER, d,
                                            sizeof(Push), 32, &t.type));
      if (!q.dense_gemv.contains(t.type))
        q.dense_gemv.emplace(t.type,
                             make_pipeline(ctx_, JR_VK_V10_DENSE_SHADER, d,
                                           sizeof(Push), 32, &t.type));
    }
    if (l < 48 && l % 4 == 3)
      q.selection =
          make_pipeline(ctx_, JR_V11_QSA_SELECT_BATCH_SHADER, d, sizeof(Push));
    if (l != 48) {
      if (l % 4 != 3 && l != 49)
        q.gdn = make_pipeline(ctx_, JR_VK_V10_GDN_SHADER, d, sizeof(Push));
      else
        q.attn = make_pipeline(ctx_, JR_VK_V10_ATTN_SHADER, d, sizeof(Push));
      auto gd = [&](VkDescriptorBufferInfo in, Buffer &out) {
        return std::vector<VkDescriptorBufferInfo>{
            info(z.ram),
            info(z.vram),
            in,
            slice(out, &out == &gu_ ? gu_stride_ : parts_stride_, row),
            slice(route_, route_stride_, row),
            info(tables_.iq2xxs),
            info(tables_.iq2xs),
            info(tables_.iq2s),
            info(tables_.iq3xxs),
            info(tables_.iq3s),
            info(tables_.signs),
            info(tables_.iq4)};
      };
      auto q8_input = info(scratch_, q8_bytes(E));
      q8_input.offset = row * w_.end * 4 + w_.round * 4;
      q.gu = make_pipeline(ctx_, shader_for_quant_type(z.gu_type),
                           gd(q8_input, gu_), 16);
      q.down = make_pipeline(ctx_, shader_for_quant_type(z.d_type),
                             gd(slice(hq_, hq_stride_, row), parts_), 16);
    }
  }
  void barrier() {
    VkMemoryBarrier b{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    b.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &b, 0,
                         nullptr, 0, nullptr);
  }
  void run(Pipeline &pipe, uint32_t x, uint32_t y, const void *push,
           uint32_t bytes, bool synchronize = true,
           uint32_t indirect_words = UINT32_MAX) {
    if (a_.profile && timing_enabled_) {
      kernel_timings_.emplace_back(nq_, active_stage_);
      vkCmdWriteTimestamp(cmd_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, queries_,
                          nq_++);
    }
    vkCmdBindPipeline(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.pipeline);
    vkCmdBindDescriptorSets(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.layout,
                            0, 1, &pipe.set, 0, nullptr);
    vkCmdPushConstants(cmd_, pipe.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, bytes,
                       push);
    if (indirect_words == UINT32_MAX)
      vkCmdDispatch(cmd_, x, y, 1);
    else
      vkCmdDispatchIndirect(cmd_, ctl_.buffer,
                            active_row_ * ctl_stride_ + indirect_words * 4);
    if (synchronize)
      barrier();
    if (a_.profile && timing_enabled_)
      vkCmdWriteTimestamp(cmd_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, queries_,
                          nq_++);
  }
  uint32_t finite_at(bool mandatory = false) {
    return a_.checks || mandatory ? 1024u : 0u;
  }
  void ops(uint32_t l, Push p, uint32_t x = 1,
           uint32_t indirect_words = UINT32_MAX) {
    if (p.op == 16 || p.op == 18)
      p.n = finite_at();
    rounded_input_ = UINT32_MAX;
    batch_input_ = UINT32_MAX;
    active_stage_ = "ops/" + std::to_string(p.op);
    // Checkpoints only read their source. The following dependent dispatch
    // supplies the execution/memory barrier before that source is overwritten.
    run(path(l).ops, x, 1, &p, sizeof(p), p.op != 19, indirect_words);
  }
  void norm(uint32_t l, uint32_t in, uint32_t out, uint32_t cols, uint32_t rows,
            const Tensor *gamma = nullptr, bool perrow = false,
            bool l2 = false) {
    Push p{1,
           in,
           out,
           gamma ? gamma->off : 0,
           cols,
           rows,
           gamma ? (perrow ? 2u : 1u) : 0u,
           l2 ? 1u : 0u};
    ops(l, p, rows);
  }
  const Tensor &named(uint32_t l, const std::string &suffix, uint32_t cols = 0,
                      uint32_t rows = 0) {
    std::string name =
        l != 48 ? "blk." + std::to_string(l) + "." + suffix : suffix;
    auto it = layers_[l].tensors.find(name);
    if (it == layers_[l].tensors.end())
      throw std::runtime_error("missing tensor " + name);
    auto &t = it->second;
    if ((cols && t.cols != cols) || (rows && t.rows != rows))
      throw std::runtime_error("wrong shape " + name);
    return t;
  }
  void prepare_round(uint32_t l, uint32_t in, uint32_t cols) {
    if (rounded_input_ != in || rounded_cols_ != cols) {
      batch_input_ = UINT32_MAX;
      Push qp{0, in, w_.round, cols};
      qp.n = finite_at();
      active_stage_ = "dense/q8";
      run(path(l).qround, cols / 32, 1, &qp, sizeof(qp));
      rounded_input_ = in;
      rounded_cols_ = cols;
    }
  }
  void gemv(uint32_t l, const std::string &name, uint32_t in, uint32_t out,
            uint32_t cols, uint32_t rows, bool independent = false,
            bool checked = false, bool subset = false) {
    auto &t = named(l, name, cols, rows);
    // BF16 projections use FP32; quantized projections share Q8_1 storage.
    if (t.type != 0 && t.type != 1 && t.type != 30) {
      prepare_round(l, in, cols);
      in = w_.round;
    }
    Push p{0, t.off, in, out, cols, rows, t.type, t.row_bytes, 0};
    if (checked)
      p.n = finite_at(name == "output.weight");
    if (subset) {
      p.m = 1;
      p.e = draft_vocab_count_;
      rows = draft_vocab_count_;
    }
    active_stage_ = "dense/" + name + "/" + std::to_string(t.type);
    if ((t.type == 0 || t.type == 30 || t.type == 1) &&
        (rows <= 64 || cols >= 8192))
      run(path(l).dense_small.at(t.type), rows, 1, &p, sizeof(p), !independent);
    else
      run(path(l).dense_gemv.at(t.type), (rows + 3) / 4, 1, &p, sizeof(p),
          !independent);
  }
  void hc_read(uint32_t l, const std::string &prefix, bool apply = false) {
    auto &n = named(l, prefix + "norm.weight", HC);
    Push np{apply ? 25u : 1u, w_.R,     w_.xn, n.off, E, 4, 2, 0,
            w_.block,         w_.inject};
    ops(l, np, 4);
    auto &down = named(l, prefix + "down.weight", HC, 320);
    const Tensor *inject =
        l != 48 ? &named(l, prefix + "inject.weight", HC, 4) : nullptr;
    if (down.type != 30 || (inject && inject->type != 30))
      throw std::runtime_error("HC down/injection must be BF16");
    Push dp{1,        down.off,
            w_.xn,    w_.lo,
            HC,       l != 48 ? 324u : 320u,
            30,       HC * 2,
            0,        inject ? inject->off : 0,
            w_.inject};
    active_stage_ = "HC/down_inject_silu";
    run(path(l).dense_small.at(30), dp.e, 1, &dp, sizeof(dp));
    auto &up = named(l, prefix + "up.weight", 320, HC);
    if (up.type != 30)
      throw std::runtime_error("HC up must be BF16");
    Push hp{0, up.off, w_.lo, w_.mixed, w_.xn};
    active_stage_ = "HC/up_mix";
    run(path(l).hc, E / 4, 1, &hp, sizeof(hp));
  }
  void check(int l, const char *stage, uint32_t in, uint32_t count,
             bool mandatory = false, bool state = false) {
    if (a_.checks || mandatory) {
      checkpoints_ = {{1024, l, stage}};
      if (state)
        ops(std::min(l, 49), {19, in, count, 1024, 1});
    }
  }
  void stamp(uint32_t bucket, bool end = false) {
    if (!timing_enabled_)
      return;
    vkCmdWriteTimestamp(cmd_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, queries_,
                        nq_);
    timings_.push_back({bucket, end ? 1u : 0u, nq_});
    nq_++;
  }
  void ple() {
    uint32_t l = 1;
    ops(l, {20, w_.pleemb}, 10);
    gemv(l, "ple_key.weight", w_.pleemb, w_.plekey, E, HC, true);
    gemv(l, "ple_value.weight", w_.pleemb, w_.pleval, E, E);
    auto &nk = named(l, "ple_norm_key.weight", HC);
    auto &nq = named(l, "ple_norm_query.weight", HC);
    auto &nc = named(l, "ple_norm_conv.weight", HC);
    norm(l, w_.plekey, w_.plekey, E, 4, &nk, true);
    norm(l, w_.R, w_.plequery, E, 4, &nq, true);
    ops(l, {17, w_.plekey, w_.plequery, w_.pleval, w_.plegated}, 4);
    norm(l, w_.plegated, w_.plenorm, E, 4, &nc, true);
    auto &cw = named(l, "ple_conv1d.weight", 4, HC);
    ops(l, {18, w_.R, w_.plenorm, 128 * 48 * 128 + HC * 3, cw.off, w_.plegated},
        40);
    check(l, "PLE", w_.R, HC);
  }
  void gdn(uint32_t l) {
    gemv(l, "attn_qkv.weight", w_.mixed, w_.qkv, E, HC, true);
    gemv(l, "ssm_alpha.weight", w_.mixed, w_.alpha, E, 48, true);
    gemv(l, "ssm_beta.weight", w_.mixed, w_.beta, E, 48, true);
    gemv(l, "attn_gate.weight", w_.mixed, w_.z, E, 6144);
    auto &cw = named(l, "ssm_conv1d.weight", 4, HC);
    ops(l, {5, w_.qkv, w_.h, 128 * 48 * 128, cw.off}, 40);
    norm(l, w_.h, w_.h, 128, 32, nullptr, false, true);
    auto &dt = named(l, "ssm_dt.bias", 48);
    auto &aa = named(l, "ssm_a", 48);
    ops(l, {6, w_.alpha, w_.beta, dt.off, aa.off});
    Push p{0, w_.h, w_.rec, w_.alpha, w_.beta, 0};
    active_stage_ = "GDN/step";
    run(path(l).gdn, 32, 48, &p, sizeof(p));
    if (a_.state_checks)
      check(l, "GDN state", 0, 128 * 48 * 128, false, true);
    auto &gn = named(l, "ssm_norm.weight", 128);
    ops(l, {7, w_.rec, w_.y, gn.off, w_.z}, 48);
    gemv(l, "ssm_out.weight", w_.y, w_.block, 6144, E, false, true);
  }
  void select_qsa(uint32_t l) {
    Push p{0, w_.scores, 0, w_.selected};
    p.l = w_.end;
    p.k = ctl_stride_ / 4;
    active_stage_ = "QSA/selection";
    run(path(l).selection, 1, 1, &p, sizeof(p), true, 16 + 2 * E + 8);
  }
  void qsa(uint32_t l) {
    gemv(l, "indexer.k_proj.weight", w_.mixed, w_.idxraw, E, 128, true);
    gemv(l, "attn_k.weight", w_.mixed, w_.kcur, E, 512, true);
    gemv(l, "attn_v.weight", w_.mixed, w_.vcur, E, 512, true);
    gemv(l, "attn_q.weight", w_.mixed, w_.qfull, E, 12288, true);
    gemv(l, "indexer.q_proj.weight", w_.mixed, w_.idxq, E, 512);
    auto &kn = named(l, "attn_k_norm.weight", 256);
    norm(l, w_.kcur, w_.kcur, 256, 2, &kn);
    ops(l, {9, w_.kcur, 256, 2});
    ops(l, {10, w_.kcur, w_.vcur, 0, a_.context});
    auto &ikn = named(l, "indexer.k_norm.weight", 128);
    ops(l, {11, w_.idxraw, 0, ikn.off, 512, 384});
    ops(l, {8, w_.qfull, w_.q}, 24);
    auto &qn = named(l, "attn_q_norm.weight", 256);
    norm(l, w_.q, w_.q, 256, 24, &qn);
    ops(l, {9, w_.q, 256, 24}, 3);
    auto &iqn = named(l, "indexer.q_norm.weight", 128);
    norm(l, w_.idxq, w_.idxq, 128, 4, &iqn);
    ops(l, {9, w_.idxq, 128, 4});
    ops(l, {12, w_.idxq, 512, 384, w_.scores}, a_.context / 4 + 1, 8);
    select_qsa(l);
    Push p{0, w_.q, w_.partial, w_.selected, 0, a_.context};
    active_stage_ = "QSA/attention";
    run(path(l).attn, 24, 65, &p, sizeof(p), true, 4);
    p = {1, w_.partial, w_.attn};
    active_stage_ = "QSA/combine";
    run(path(l).attn, 24, 1, &p, sizeof(p));
    ops(l, {14, w_.attn, w_.qfull, w_.y}, 24);
    gemv(l, "attn_output.weight", w_.y, w_.block, 6144, E, false, true);
  }
  void moe(uint32_t l) {
    stamp(2);
    gemv(l, "ffn_gate_inp.weight", w_.mixed, w_.scores, E, 512, false, true);
    Push rp{0, w_.scores, 16 + l * 16};
    active_stage_ = "router/top10";
    run(path(l).router, 1, 1, &rp, sizeof(rp));
    check(l, "router", w_.scores, 512);
    stamp(2, true);
    stamp(1);
    prepare_round(l, w_.mixed, E);
    stamp(1, true);
    std::array<uint32_t, 4> gp{K, E, FF, 0};
    std::array<uint32_t, 2> sp{K, FF};
    if (timing_enabled_) {
      // Sampled commands keep disjoint buckets for the inherited V9 budget.
      stamp(3);
      active_stage_ = "MoE/gate_up";
      run(path(l).gu, 2 * FF / 4, K, gp.data(), 16);
      active_stage_ = "MoE/swiglu";
      run(swiglu_[active_row_], FF / 32, K, sp.data(), 8);
      gp[3] = 1;
      active_stage_ = "MoE/down";
      run(path(l).down, E / 4, K, gp.data(), 16);
      sp = {K, E};
      active_stage_ = "MoE/combine";
      run(combine_[active_row_], 10, 1, sp.data(), 8);
      stamp(3, true);
      stamp(1);
      gemv(l, "ffn_gate_shexp.weight", w_.mixed, w_.shg, E, FF, true);
      gemv(l, "ffn_up_shexp.weight", w_.mixed, w_.shu, E, FF);
      ops(l, {15, w_.shg, w_.shu, w_.shh}, 3);
      gemv(l, "ffn_down_shexp.weight", w_.shh, w_.sho, FF, E);
      gemv(l, "ffn_gate_inp_shexp.weight", w_.mixed, w_.shgate, E, 1);
      ops(l, {16, w_.moe, w_.sho, w_.block, w_.shgate}, 10);
      check(l, "MoE/shared", w_.block, E);
      stamp(1, true);
    } else {
      // Routed and shared experts read the same immutable input. Overlap their
      // independent projections, then join only at true dependencies.
      run(path(l).gu, 2 * FF / 4, K, gp.data(), 16, false);
      gemv(l, "ffn_gate_shexp.weight", w_.mixed, w_.shg, E, FF, true);
      gemv(l, "ffn_up_shexp.weight", w_.mixed, w_.shu, E, FF);
      run(swiglu_[active_row_], FF / 32, K, sp.data(), 8, false);
      ops(l, {15, w_.shg, w_.shu, w_.shh}, 3);
      gemv(l, "ffn_down_shexp.weight", w_.shh, w_.sho, FF, E, true);
      gemv(l, "ffn_gate_inp_shexp.weight", w_.mixed, w_.shgate, E, 1, true);
      gp[3] = 1;
      run(path(l).down, E / 4, K, gp.data(), 16);
      sp = {K, E};
      run(combine_[active_row_], 10, 1, sp.data(), 8);
      ops(l, {16, w_.moe, w_.sho, w_.block, w_.shgate}, 10);
      check(l, "MoE/shared", w_.block, E);
    }
  }
  void record() {
    nq_ = 0;
    timings_.clear();
    kernel_timings_.clear();
    checkpoints_.clear();
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    vk_check(vkBeginCommandBuffer(cmd_, &bi), "begin full token");
    if (timing_enabled_)
      vkCmdResetQueryPool(cmd_, queries_, 0, 8192);
    VkMemoryBarrier host{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    host.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
    host.dstAccessMask =
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
    vkCmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_HOST_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                             VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
                         0, 1, &host, 0, nullptr, 0, nullptr);
    stamp(0);
    Push embedding_push{0, w_.R, 0};
    embedding_push.n = finite_at();
    active_stage_ = "embedding/imported_RAM";
    run(embedding_[active_row_], 40, 1, &embedding_push,
        sizeof(embedding_push));
    check(0, "embedding", w_.R, HC);
    stamp(0, true);
    for (uint32_t l = 0; l < 48; l++) {
      if (l == 1) {
        stamp(4);
        ops(l, {4, w_.R, w_.block, w_.inject}, 40);
        ple();
        stamp(4, true);
      }
      stamp(1);
      hc_read(l, "hc_attn_", l > 0 && l != 1);
      if (l % 4 == 3)
        qsa(l);
      else
        gdn(l);
      check(l, l % 4 == 3 ? "attention" : "GDN", w_.block, E);
      hc_read(l, "hc_ffn_", true);
      stamp(1, true);
      moe(l);
    }
    stamp(5);
    hc_read(48, "output_hc_", true);
    gemv(48, "output.weight", w_.mixed, w_.head, E, VOCAB, false, true);
    check(48, "head", w_.head, VOCAB, true);
    ops(48, {22, w_.head, VOCAB});
    if (head_host_.buffer) {
      VkMemoryBarrier ready{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
      ready.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      ready.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
      vkCmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &ready, 0,
                           nullptr, 0, nullptr);
      VkBufferCopy c{uint64_t(w_.head) * 4, 0, VOCAB * 4ull};
      vkCmdCopyBuffer(cmd_, scratch_.buffer, head_host_.buffer, 1, &c);
    }
    stamp(5, true);
    VkMemoryBarrier read{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    read.srcAccessMask =
        VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    read.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(
        cmd_,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &read, 0, nullptr, 0, nullptr);
    vk_check(vkEndCommandBuffer(cmd_), "end full token");
    timestamps_.resize(nq_);
  }
  void dequant_row(const uint8_t *data, uint32_t type, float *out, uint32_t n) {
    if (type == 20) {
      for (uint32_t i = 0; i < n; i += 32)
        strata::dequantize_iq4_nl(data + (i / 32) * 18, out + i);
      return;
    }
    if (type == 21) {
      for (uint32_t i = 0; i < n; i++) {
        auto off = (i / 256) * 110, j = i % 256, ib = j / 32, il = (j % 32) / 8,
             q = (j % 8) / 4;
        auto *b = data + off;
        uint32_t gi = b[2 + ib * 8 + il * 2 + q] |
                      (((b[66 + ib] >> (il * 2 + q)) & 1) << 8);
        int sym = int((iq3s_grid[gi] >> ((j % 4) * 8)) & 255);
        if ((b[74 + ib * 4 + il] >> (j % 8)) & 1)
          sym = -sym;
        out[i] = strata::fp16_to_fp32(strata::read_u16(b)) *
                 float(1 + 2 * ((b[106 + ib / 2] >> ((ib % 2) * 4)) & 15)) *
                 sym;
      }
      return;
    }
    throw std::runtime_error("unsupported embedding type");
  }
  void stage(uint32_t token) {
    auto *c = reinterpret_cast<uint32_t *>(static_cast<uint8_t *>(ctl_.mapped) +
                                           active_row_ * ctl_stride_);
    c[0] = position_;
    c[1] = token;
    const uint32_t count = position_ + 1;
    c[4] = count > 512 ? 3 : 24;
    c[5] = (std::min(count, 2051u) + 31) / 32;
    c[6] = 1;
    // QSA consumes all keys below the selection limit; scoring is unnecessary.
    c[8] = count > 2051 ? count / 4 + 1 : 0;
    c[9] = 1;
    c[10] = 1;
    c[12] = ((count <= 2051 ? count : count / 4 + 1) + 255) / 256;
    c[13] = 1;
    c[14] = 1;
    constexpr uint32_t select_dispatch = 16 + 2 * E + 8;
    c[select_dispatch] =
        count <= 2051 ? (count + 255) / 256 : (count / 4 + 8) / 8;
    c[select_dispatch + 1] = 1;
    c[select_dispatch + 2] = 1;

    std::array<uint64_t, 3> ctx{token, 248044, 248044};
    bool cut = false;
    for (int i = 1; i < 3; i++) {
      int64_t v = previous_count_ >= size_t(i)
                      ? int64_t(previous_[previous_count_ - i])
                      : int64_t(-1);
      cut = cut || v < 0 || v == 248044;
      ctx[i] = cut ? 248044 : v;
    }
    uint64_t hash = ctx[0] * multipliers_[0];
    for (int n = 1; n < 3; n++) {
      hash ^= ctx[n] * multipliers_[n];
      for (int j = 0; j < 8; j++) {
        int h = (n - 1) * 8 + j;
        uint32_t r = hash % vocab_[h] + offsets_[h];
        size_t cache_slot = r % ple_cache_rows;
        if (plecache_ids_[cache_slot] != r) {
          std::array<uint8_t, 90> raw{};
          size_t got = 0;
          while (got < raw.size()) {
            ssize_t k = pread(plefd_, raw.data() + got, raw.size() - got,
                              pleoff_ + uint64_t(r) * 90 + got);
            if (k <= 0)
              throw std::runtime_error("PLE short read");
            got += k;
          }
          dequant_row(raw.data(), 20, plecache_[cache_slot].data(), 160);
          plecache_ids_[cache_slot] = r;
        }
        std::memcpy(c + 16 + E + h * 160, plecache_[cache_slot].data(),
                    160 * 4);
      }
    }
  }
  void crash(const Checkpoint &cp, const std::string &reason = "nonfinite") {
    fs::path path = a_.dump.empty() ? "v11-crash.json"
                                    : fs::path(a_.dump) / "v11-crash.json";
    auto *d = reinterpret_cast<float *>(static_cast<uint8_t *>(diag_.mapped) +
                                        active_row_ * 65536);
    std::ofstream f(path);
    f << "{\"position\":" << position_
      << ",\"token\":" << static_cast<uint32_t *>(ctl_.mapped)[1]
      << ",\"layer\":" << cp.layer << ",\"stage\":\"" << cp.stage
      << "\",\"previous\":[";
    for (size_t i = 0; i < previous_count_; i++)
      f << (i ? "," : "") << previous_[i];
    auto number = [&](float value) {
      if (std::isfinite(value))
        f << value;
      else
        f << "null";
    };
    f << "],\"reason\":" << std::quoted(reason) << ",\"min\":";
    number(d[cp.offset]);
    f << ",\"max\":";
    number(d[cp.offset + 1]);
    f << ",\"norm\":";
    number(d[cp.offset + 2]);
    f << ",\"nonfinite\":";
    number(d[cp.offset + 3]);
    f << "}\n";
  }

#include "v11_mtp_runtime.inc"
public:
  explicit Runtime(const Options &a)
      : a_(a), ctx_(b60_selector()), model_(strata::GgufModel::open(a.native)),
        plan_(read_cache_plan(a.residency)) {
    if (ctx_.caps().vendor_id != 0x8086 || ctx_.caps().device_id != 0xe211)
      throw std::runtime_error("V11 requires Intel B60 8086:e211");
    if (!ctx_.caps().ext_external_memory_host ||
        !ctx_.caps().intdot_4x8_packed_signed_accelerated)
      throw std::runtime_error(
          "V11 requires imported host memory and hardware packed integer dot");
    verify_geometry();
    std::ifstream idx(fs::path(a.pack) / "index.txt");
    std::string identity;
    std::getline(idx, identity);
    if (!idx || identity.find("native experts") == std::string::npos)
      throw std::runtime_error("not a native IQ pack");
    // Verify native expert records refer to exactly these GGUF payloads,
    // including shard names.
    std::ifstream ef(fs::path(a.pack) / "native_experts.txt");
    if (!ef)
      throw std::runtime_error("native_experts.txt missing");
    std::string line;
    uint32_t count = 0;
    while (std::getline(ef, line)) {
      if (line.empty() || line[0] == '#')
        continue;
      std::istringstream in(line);
      int l, gt, dt;
      uint64_t off, bytes, go, uo, doff;
      std::string shard;
      if (!(in >> l >> gt >> dt >> off >> bytes >> go >> uo >> doff) ||
          l != int(count) || l >= 48)
        throw std::runtime_error("invalid native record");
      in >> shard;
      const char *roles[] = {"ffn_gate_exps.weight", "ffn_up_exps.weight",
                             "ffn_down_exps.weight"};
      uint64_t offsets[] = {go, uo, doff};
      uint64_t expected_bytes = 0;
      for (int r = 0; r < 3; r++) {
        size_t sh;
        auto &t = tensor("blk." + std::to_string(l) + "." + roles[r], sh);
        auto &file = model_.shard(sh);
        if (t.type != uint32_t(r < 2 ? gt : dt) ||
            file.data_start() + t.offset != offsets[r] ||
            (!shard.empty() && fs::path(file.path()).filename() != shard))
          throw std::runtime_error("native pack/GGUF identity mismatch");
        expected_bytes += strata::tensor_payload_bytes(t) / 512;
      }
      if (expected_bytes != bytes)
        throw std::runtime_error("native blob size mismatch");
      count++;
    }
    if (count != 48)
      throw std::runtime_error("native pack does not contain 48 layers");
    std::cerr << "PCI device: 8086:e211\nVulkan device: "
              << ctx_.caps().device_name
              << "\nGeometry: 48 layers, 2560 embedding, HC 4x320, 512 "
                 "experts, top-10\nContext: "
              << a.context
              << " KV: FP16; greedy; single request\nV9 resident experts: "
              << plan_.n_resident << " / 24576, "
              << double(plan_.used_bytes) / (1ull << 30)
              << " GiB\nPack identity: " << a.pack << " / " << identity
              << "\nMTP spec: " << a_.spec << "\n";
    tables_ = upload_tables(ctx_);
    // The dense decoder retains the original IQ3 prefix and appends a 1 KiB
    // exact IQ4 pair table. Routed expert descriptors keep their V9 tables.
    static_assert(sizeof(iq3s_grid) / sizeof(iq3s_grid[0]) == 512);
    std::vector<uint32_t> dense_grid(iq3s_grid, iq3s_grid + 512);
    for (uint32_t code = 0; code < 256; code++)
      dense_grid.push_back((uint32_t(kvalues_iq4nl[code & 15]) & 255) |
                           ((uint32_t(kvalues_iq4nl[code >> 4]) & 255) << 8));
    dense_grid_ = upload_device(ctx_, dense_grid.data(), dense_grid.size() * 4);
    rows_ = a_.spec ? 6 : 1;
    ctl_stride_ = align_up((16 + 2 * E) * 4, 256);
    scratch_ = buffer(w_.end * 4 * rows_);
    ctl_ = buffer(ctl_stride_ * rows_, true, true);
    diag_ = buffer(65536 * rows_, true);
    route_ = buffer(route_stride_ * rows_);
    weights_ = buffer(weights_stride_ * rows_);
    gu_ = buffer(gu_stride_ * rows_);
    hq_ = buffer(hq_stride_ * rows_);
    parts_ = buffer(parts_stride_ * rows_);
    dummy_ = buffer(4);
    if (!a.logits.empty())
      head_host_ = buffer(VOCAB * 4, true);
    embed_ = &tensor("token_embd.weight", embed_shard_);
    if (embed_->shape != std::vector<uint64_t>{E, VOCAB} || embed_->type != 21)
      throw std::runtime_error("unsupported token embedding");
    {
      const auto *raw = model_.shard(embed_shard_).tensor_data(*embed_);
      std::vector<uint8_t> data(raw,
                                raw + strata::tensor_payload_bytes(*embed_));
      embedding_ram_ = make_imported_host(ctx_, data);
      release_source_pages(model_.shard(embed_shard_), *embed_);
      std::cerr << "Embedding: imported host memory, " << data.size()
                << " B; head remains device-local\n";
    }
    load_ple();
    uint64_t resident_bytes = 0;
    for (uint32_t l = 0; l < 49; l++) {
      load_dense(l);
      auto &z = layers_[l];
      if (l < 48) {
        load_experts(l);
        resident_bytes += z.vram.size;
        z.state = buffer(
            l % 4 != 3 ? (128 * 48 * 128 + HC * 3 + (l == 1 ? HC * 9 : 0)) * 4
                       : ((a.context / 4 + 2) * 128 + 512) * 4);
        z.kv = buffer(l % 4 == 3 ? uint64_t(a.context) * 512 * 2 * 2 : 4);
      } else {
        z.state = buffer(4);
        z.kv = buffer(4);
        z.plan = buffer(4);
      }

      if (l % 8 == 7 || l == 48) {
        std::cerr << "Loaded " << l + 1 << " / 49 stages\n";
        memory_report("load layer " + std::to_string(l));
      }
    }
    if (resident_bytes < plan_.used_bytes ||
        resident_bytes > plan_.used_bytes + 48 * 64)
      throw std::runtime_error("resident byte total differs from V9 plan");
    if (a_.spec)
      load_mtp();
    else {
      layers_[49].state = buffer(4);
      layers_[49].kv = buffer(4);
    }
    for (uint32_t l = 0; l < 48; l++) {
      snapshot_offset_[l] = snapshot_stride_;
      snapshot_bytes_[l] = l % 4 == 3 ? 512 * 4 : layers_[l].state.size;
      snapshot_stride_ += align_up(snapshot_bytes_[l], 256);
    }
    // The fifth row is the all-accepted final state and never needs rollback.
    snapshots_ = buffer(a_.spec ? snapshot_stride_ * 4 : 4);
    if (!draft_vocab_.buffer)
      draft_vocab_ = buffer(4);
    if (a_.spec)
      mtp_partial_ = buffer(24ull * 256 * 258 * 4);
    for (uint32_t row = 0; row < rows_; row++) {
      for (uint32_t l = 0; l < (a_.spec ? 50u : 49u); l++)
        pipelines(l, row);
      auto ed = desc(48, row);
      ed[0] = info(embedding_ram_);
      embedding_[row] =
          make_pipeline(ctx_, JR_VK_V10_OPS_SHADER, ed, sizeof(Push));
      swiglu_[row] = make_pipeline(
          ctx_, JR_VK_V5_GROUPED_SWIGLU_SHADER,
          {slice(gu_, gu_stride_, row), slice(hq_, hq_stride_, row)}, 8);
      auto output = info(scratch_, E * 4);
      output.offset = row * w_.end * 4 + w_.moe * 4;
      combine_[row] =
          make_pipeline(ctx_, JR_VK_V5_COMBINE_SHADER,
                        {slice(parts_, parts_stride_, row),
                         slice(weights_, weights_stride_, row), output},
                        8);
    }
    if (a_.spec) {
      auto md = desc(49, 5);
      md.push_back(info(mtp_partial_));
      layers_[49].paths[5].attn =
          make_pipeline(ctx_, JR_V11_MTP_ATTN_SHADER, md, sizeof(Push));
    }
    auto staged_weights = info(snapshots_, snapshot_stride_);
    staged_weights.offset = 3 * snapshot_stride_;
    auto staged_meta = info(scratch_, 50 * 8 * 4);
    staged_meta.offset = uint64_t(w_.xn) * 4;
    if (a_.spec) {
      batch_swiglu_ = make_pipeline(ctx_, JR_V11_SWIGLU_BATCH_SHADER,
                                    {info(gu_), info(hq_), info(scratch_)}, 32);
      batch_combine_ = make_pipeline(
          ctx_, JR_V11_COMBINE_BATCH_SHADER,
          {info(parts_), info(weights_), info(scratch_), info(diag_)}, 40);
      group_meta_ = buffer(50 * 8 * 4);
      group_slots_ = buffer(520 * 4, false, true);
      group_routes_ = make_pipeline(
          ctx_, JR_V11_VERIFY_PREFETCH_GROUP_SHADER,
          {info(route_), info(group_meta_), info(group_slots_),
           staged_meta}, 16);
    }
    if (a_.spec)
      for (uint32_t l = 0; l < 49; l++) {
        auto bd = desc(l);
        bd[1] = info(scratch_);
        bd[2] = info(ctl_);
        bd[5] = info(diag_);
        bd[8] = info(route_);
        bd[9] = info(weights_);
        if (l < 48)
          layers_[l].batch_router =
              make_pipeline(ctx_, JR_V11_ROUTER_BATCH_SHADER, bd, sizeof(Push));
        if (l % 4 == 3 && l < 48) {
          layers_[l].batch_selection = make_pipeline(
              ctx_, JR_V11_QSA_SELECT_BATCH_SHADER, bd, sizeof(Push));
          layers_[l].batch_qsa_prepare = make_pipeline(
              ctx_, JR_V11_VERIFY_QSA_PREPARE_SHADER, bd, sizeof(Push));
          layers_[l].batch_qsa_score = make_pipeline(
              ctx_, JR_V11_VERIFY_QSA_SCORE_SHADER, bd, sizeof(Push), 32);
          layers_[l].batch_qsa_ops = make_pipeline(
              ctx_, JR_V11_QSA_OPS_BATCH_SHADER, bd, sizeof(Push));
          layers_[l].batch_attn =
              make_pipeline(ctx_, JR_V11_ATTN_BATCH_SHADER, bd, sizeof(Push));
          layers_[l].batch_attn_tile = make_pipeline(
              ctx_, JR_V11_VERIFY_ATTN_TILE_SHADER, bd, sizeof(Push));
        }
        layers_[l].batch_norm =
            make_pipeline(ctx_, JR_V11_NORM_BATCH_SHADER, bd, sizeof(Push));
        layers_[l].batch_qround =
            make_pipeline(ctx_, JR_V11_QROUND_BATCH_SHADER, bd, sizeof(Push));
        for (uint32_t n = 1; n <= 5; n++) {
          for (auto &[name, t] : layers_[l].tensors) {
            uint32_t format = t.type | (n << 8);
            if (!layers_[l].batch_dense[n].contains(t.type))
              layers_[l].batch_dense[n].emplace(
                  t.type,
                  make_pipeline(ctx_, JR_V11_DENSE_BATCH_SHADER, bd,
                                sizeof(Push), t.type == 13 ? 32 : 16, &format));
            if ((t.type == 0 || t.type == 30 || t.type == 1) &&
                !layers_[l].batch_small[n].contains(t.type))
              layers_[l].batch_small[n].emplace(
                  t.type, make_pipeline(ctx_, JR_V11_SMALL_BATCH_SHADER, bd,
                                        sizeof(Push), 32, &format));
          }
          uint32_t hc_format = 30 | (n << 8);
          layers_[l].batch_hc_down[n] =
              make_pipeline(ctx_, JR_V11_VERIFY_HC_DOWN_SHADER, bd,
                            sizeof(Push), 32, &hc_format);
          layers_[l].batch_hc[n] = make_pipeline(
              ctx_, JR_V11_VERIFY_HC_OUTPUT_SHADER, bd, sizeof(Push), 32, &n);
        }
        if (l % 4 != 3 && l < 48) {
          layers_[l].batch_gdn_output = make_pipeline(
              ctx_, JR_V11_VERIFY_GDN_OUTPUT_SHADER, bd, sizeof(Push));
          layers_[l].batch_gdn_prepare = make_pipeline(
              ctx_, JR_V11_GDN_PREPARE_BATCH_SHADER, bd, sizeof(Push), 32);
          layers_[l].batch_gdn =
              make_pipeline(ctx_, JR_V11_GDN_BATCH_SHADER, bd, sizeof(Push));
        }
        if (l < 48) {
          auto &z = layers_[l];
          uint64_t gu_bytes = uint64_t(FF) * row_bytes_for(z.gu_type, E),
                   down_bytes = uint64_t(E) * row_bytes_for(z.d_type, FF);
          if (40 * (2 * gu_bytes + down_bytes) > snapshot_stride_ ||
              gu_bytes % 16 || down_bytes % 16)
            throw std::runtime_error(
                "native verification staging exceeds unused snapshot");
          z.verify_prefetch =
              make_pipeline(ctx_, JR_V11_VERIFY_PREFETCH_SHADER,
                            {info(z.ram), info(group_meta_), info(group_slots_),
                             staged_weights, staged_meta},
                            8);
          auto gd = [&](VkDescriptorBufferInfo input, Buffer &output,
                        bool prefetch) {
            return std::vector<VkDescriptorBufferInfo>{
                prefetch ? staged_weights : info(z.ram),
                info(z.vram),
                input,
                info(output),
                prefetch ? staged_meta : info(group_meta_),
                info(tables_.iq2xxs),
                info(tables_.iq2xs),
                info(tables_.iq2s),
                info(tables_.iq3xxs),
                info(tables_.iq3s),
                info(tables_.signs),
                info(tables_.iq4),
                info(group_slots_),
                info(scratch_)};
          };
          auto input = info(scratch_, scratch_.size - w_.round * 4);
          input.offset = w_.round * 4;
          for (uint32_t n = 1; n <= 5; n++) {
            // Two-position IQ3 windows use fewer registers. Real-weight checks
            // measured a 5-8% gate/up gain with identical FP32 output bits.
            uint32_t gu_columns =
                n == 3
                    ? 3
                    : (n == 2 && (z.gu_type == 18 || z.gu_type == 21) ? 2 : 5);
            gu_columns |= n << 8;
            uint32_t down_format = 5 | (n << 8);
            uint32_t gu_subgroup =
                (n == 3 &&
                 (z.gu_type == 16 || z.gu_type == 17 || z.gu_type == 22)) ||
                        (n == 2 && (z.gu_type == 18 || z.gu_type == 21))
                    ? 32
                    : 16;
            z.batch_gu[n] = make_pipeline(
                ctx_,
                batch_expert_shader(z.gu_type, batch_gate_rows(z.gu_type, n)),
                gd(input, gu_, n <= 4), 32, gu_subgroup, &gu_columns);
            z.batch_down[n] = make_pipeline(ctx_, batch_expert_shader(z.d_type),
                                            gd(info(hq_), parts_, n <= 4), 32,
                                            32, &down_format);
          }
        }
      }
    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pci.queueFamilyIndex = ctx_.queue_family();
    vk_check(vkCreateCommandPool(ctx_.device(), &pci, nullptr, &pool_),
             "create token pool");
    VkCommandBufferAllocateInfo cai{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = pool_;
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 2;
    VkCommandBuffer commands[2];
    vk_check(vkAllocateCommandBuffers(ctx_.device(), &cai, commands),
             "allocate token commands");
    cmd_ = commands[0];
    fast_cmd_ = commands[1];
    VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    vk_check(vkCreateFence(ctx_.device(), &fi, nullptr, &fence_),
             "create token fence");
    VkQueryPoolCreateInfo qi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    qi.queryType = VK_QUERY_TYPE_TIMESTAMP;
    qi.queryCount = 8192;
    vk_check(vkCreateQueryPool(ctx_.device(), &qi, nullptr, &queries_),
             "create token timing queries");
    VkPhysicalDeviceProperties prop;
    vkGetPhysicalDeviceProperties(ctx_.physical_device(), &prop);
    timestamp_period_ = prop.limits.timestampPeriod;
    reset();
    auto timed_cmd = cmd_;
    cmd_ = fast_cmd_;
    timing_enabled_ = false;
    record();
    cmd_ = timed_cmd;
    timing_enabled_ = true;
    record();
    std::cerr << "Full-token commands ready: " << nq_
              << " timestamps; fixed scratch " << w_.end * 4 << " B; checks "
              << a.checks << "\n";
    if (a_.spec)
      initialize_mtp_commands();
  }
  ~Runtime() {
    if (ctx_.device())
      vkDeviceWaitIdle(ctx_.device());
    if (plefd_ >= 0)
      close(plefd_);
    if (batch_queries_)
      vkDestroyQueryPool(ctx_.device(), batch_queries_, nullptr);
    if (queries_)
      vkDestroyQueryPool(ctx_.device(), queries_, nullptr);
    if (fence_)
      vkDestroyFence(ctx_.device(), fence_, nullptr);
    if (pool_)
      vkDestroyCommandPool(ctx_.device(), pool_, nullptr);
  }
  void reset() {
    position_ = 0;
    previous_count_ = 0;
    active_row_ = 0;
    metrics_ = {};
    metrics_.adaptive_max_drafts = a_.spec;
    test_override_count_ = 0;
    std::memset(ctl_.mapped, 0, ctl_.size);
    std::memset(diag_.mapped, 0, diag_.size);
    VkCommandPoolCreateInfo pi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pi.queueFamilyIndex = ctx_.queue_family();
    VkCommandPool p;
    vk_check(vkCreateCommandPool(ctx_.device(), &pi, nullptr, &p),
             "reset pool");
    VkCommandBufferAllocateInfo ai{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = p;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkCommandBuffer c;
    vk_check(vkAllocateCommandBuffers(ctx_.device(), &ai, &c),
             "reset allocate");
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    vk_check(vkBeginCommandBuffer(c, &bi), "reset begin");
    vkCmdFillBuffer(c, scratch_.buffer, 0, VK_WHOLE_SIZE, 0);
    for (auto &l : layers_) {
      if (l.state.buffer)
        vkCmdFillBuffer(c, l.state.buffer, 0, VK_WHOLE_SIZE, 0);
      if (l.kv.buffer)
        vkCmdFillBuffer(c, l.kv.buffer, 0, VK_WHOLE_SIZE, 0);
    }
    VkMemoryBarrier b{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(c, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &b, 0,
                         nullptr, 0, nullptr);
    vk_check(vkEndCommandBuffer(c), "reset end");
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence f;
    vk_check(vkCreateFence(ctx_.device(), &fci, nullptr, &f), "reset fence");
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &c;
    vk_check(vkQueueSubmit(ctx_.queue(), 1, &si, f), "reset submit");
    vk_check(vkWaitForFences(ctx_.device(), 1, &f, VK_TRUE, UINT64_MAX),
             "reset wait");
    vkDestroyFence(ctx_.device(), f, nullptr);
    vkDestroyCommandPool(ctx_.device(), p, nullptr);
  }
  TokenResult token(uint32_t id, bool mtp_seeded = false) {
    if (id >= VOCAB || position_ >= a_.context)
      throw std::runtime_error("token or context out of range");
    auto begin = std::chrono::steady_clock::now();
    active_row_ = 0;
    stage(id);
    if (a_.spec && position_ > 0 && !mtp_seeded)
      stage_mtp(position_ - 1, id);
    vk_check(vkResetFences(ctx_.device(), 1, &fence_), "reset token fence");
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = a_.spec && position_ > 0 && !mtp_seeded ? 2 : 1;
    bool timed = position_ % a_.stats_every == 0;
    auto selected_cmd = timed ? cmd_ : fast_cmd_;
    VkCommandBuffer pair[] = {catchup_cmd_[0], selected_cmd};
    si.pCommandBuffers = si.commandBufferCount == 2 ? pair : &selected_cmd;
    vk_check(vkQueueSubmit(ctx_.queue(), 1, &si, fence_), "submit full token");
    vk_check(
        vkWaitForFences(ctx_.device(), 1, &fence_, VK_TRUE, 120000000000ull),
        "wait full token");
    auto *d = reinterpret_cast<float *>(static_cast<uint8_t *>(diag_.mapped) +
                                        active_row_ * 65536);
    for (auto &cp : checkpoints_)
      if (d[cp.offset + 3] != 0 || !std::isfinite(d[cp.offset + 2])) {
        crash(cp);
        throw std::runtime_error("nonfinite at position " +
                                 std::to_string(position_) + " layer " +
                                 std::to_string(cp.layer) + " " + cp.stage);
      }
    TokenResult r{std::bit_cast<uint32_t>(d[0]), d[1], 0, {}};
    if (r.token >= VOCAB || !std::isfinite(r.logit))
      throw std::runtime_error("invalid sampler result");
    if (timed) {
      vk_check(vkGetQueryPoolResults(ctx_.device(), queries_, 0, nq_,
                                     timestamps_.size() * 8, timestamps_.data(),
                                     8, VK_QUERY_RESULT_64_BIT),
               "read token timestamps");
      std::array<uint64_t, 6> begins{};
      for (auto mark : timings_) {
        auto [bucket, end, i] = mark;
        if (!end)
          begins[bucket] = timestamps_[i];
        else
          r.gpu_ms[bucket] += double(timestamps_[i] - begins[bucket]) *
                              timestamp_period_ * 1e-6;
      }
    }
    if (a_.profile && timed) {
      std::map<std::string, double> costs;
      for (auto &[i, name] : kernel_timings_)
        costs[name] += double(timestamps_[i + 1] - timestamps_[i]) *
                       timestamp_period_ * 1e-6;
      std::vector<std::pair<double, std::string>> sorted;
      for (auto &[name, ms] : costs)
        sorted.emplace_back(ms, name);
      std::sort(sorted.rbegin(), sorted.rend());
      for (size_t i = 0; i < std::min<size_t>(30, sorted.size()); i++)
        std::cerr << "PROFILE " << sorted[i].second << " " << sorted[i].first
                  << " ms\n";
    }
    for (uint32_t l = 0; l < 48; l++)
      for (uint32_t k = 0; k < 10; k++) {
        auto eid = std::bit_cast<uint32_t>(d[16 + l * 16 + k]);
        if (eid >= 512)
          throw std::runtime_error("router ID out of bounds");
        route_counts_[l * 512 + eid]++;
        route_hits_ += plan_.slot[l * 512 + eid] >= 0;
        route_count_++;
      }
    if ((position_ + 1) % a_.stats_every == 0) {
      std::cerr << "V11 real-route VRAM hit rate: "
                << 100.0 * route_hits_ / route_count_ << "%\n";
      memory_report("position " + std::to_string(position_ + 1));
    }
    if (previous_count_ == 16) {
      std::memmove(previous_.data(), previous_.data() + 1, 15 * 4);
      previous_[15] = id;
    } else
      previous_[previous_count_++] = id;
    position_++;
    r.total_ms = std::chrono::duration<double, std::milli>(
                     std::chrono::steady_clock::now() - begin)
                     .count();
    if (head_host_.buffer)
      write_f32(a_.logits, static_cast<float *>(head_host_.mapped), VOCAB);
    return r;
  }
  void export_profile() {
    if (a_.route_profile.empty())
      return;
    std::vector<uint32_t> ranked(48 * 512);
    std::iota(ranked.begin(), ranked.end(), 0);
    std::stable_sort(ranked.begin(), ranked.end(), [&](uint32_t x, uint32_t y) {
      if (route_counts_[x] != route_counts_[y])
        return route_counts_[x] > route_counts_[y];
      bool xr = plan_.slot[x] >= 0, yr = plan_.slot[y] >= 0;
      if (xr != yr)
        return xr;
      return x < y;
    });
    std::ofstream f(a_.route_profile, std::ios::binary);
    if (!f)
      throw std::runtime_error("cannot create route profile");
    f.write("STRP", 4);
    uint32_t header[] = {1, 48, 512, plan_.n_resident, 48 * 512};
    f.write(reinterpret_cast<char *>(header), sizeof(header));
    for (auto id : ranked) {
      uint16_t pair[] = {uint16_t(id / 512), uint16_t(id % 512)};
      f.write(reinterpret_cast<char *>(pair), 4);
    }
    if (!f)
      throw std::runtime_error("route profile write failed");
    std::cerr << "V11 real-route STRP profile: " << a_.route_profile << ", "
              << route_count_ << " selections\n";
  }
  void worker() {
    std::cout << "READY\n" << std::flush;
    std::string line;
    while (std::getline(std::cin, line)) {
      try {
        if (line == "RESET") {
          reset();
          std::cout << "OK\n";
        } else if (line == "STATS") {
          std::cout << "STATS " << metrics_json() << "\n";
        } else if (line == "QUIT")
          break;
        else if (line.starts_with("CONFIDENCE ")) {
          std::istringstream in(line.substr(11));
          float p;
          std::string extra;
          if (!(in >> p) || (in >> extra) || !std::isfinite(p) || p < 0 ||
              p > 1)
            throw std::runtime_error("invalid confidence");
          a_.min_p = p;
          std::cout << "OK\n";
        } else if (line.starts_with("REJECT ")) {
          std::istringstream in(line.substr(7));
          int value;
          std::string extra;
          if (!(in >> value) || (in >> extra) || value < -1 || value > 3)
            throw std::runtime_error("invalid test rejection index");
          a_.force_reject = value;
          std::cout << "OK\n";
        } else if (line.starts_with("OVERRIDE ")) {
          std::istringstream in(line.substr(9));
          test_override_count_ = 0;
          uint32_t id;
          while (in >> id) {
            if (test_override_count_ == 4 || id >= VOCAB)
              throw std::runtime_error("invalid test proposal override");
            test_override_[test_override_count_++] = id;
          }
          if (!in.eof())
            throw std::runtime_error("invalid override");
          std::cout << "OK\n";
        } else if (line.starts_with("ROUND ")) {
          std::istringstream in(line);
          std::string op, extra;
          uint32_t id, limit;
          if (!(in >> op >> id >> limit) || (in >> extra))
            throw std::runtime_error("expected ROUND id limit");
          round(id, limit);
        } else {
          std::istringstream in(line);
          std::string op;
          uint32_t id;
          std::string extra;
          if (!(in >> op >> id) || op != "TOKEN" || (in >> extra))
            throw std::runtime_error("expected TOKEN <id>");
          auto r = token(id);
          std::cout << "TOKEN " << r.token << " " << std::setprecision(9)
                    << r.logit << " " << r.total_ms;
          for (auto x : r.gpu_ms)
            std::cout << " " << x;
          std::cout << "\n";
        }
      } catch (const std::exception &e) {
        if (std::string(e.what()).find("nonfinite at position") != 0)
          crash({1024, -1, "worker/device"}, e.what());
        std::cout << "ERROR " << e.what() << "\n";
        std::cout.flush();
        return;
      }
      std::cout.flush();
    }
  }
  void ids() {
    auto bytes = read_bytes(a_.ids);
    if (bytes.empty() || bytes.size() % 4)
      throw std::runtime_error("--ids must be nonempty little-endian int32");
    if (bytes.size() / 4 + a_.max_new > a_.context)
      throw std::runtime_error("prompt plus generation exceeds context");
    TokenResult r{};
    for (size_t i = 0; i < bytes.size(); i += 4) {
      uint32_t id;
      std::memcpy(&id, bytes.data() + i, 4);
      r = token(id);
    }
    double total = 0;
    uint32_t emitted = 0, current = r.token, repeated = 0,
             previous = UINT32_MAX;
    auto output = [&](uint32_t id) {
      repeated = id == previous ? repeated + 1 : 1;
      previous = id;
      if (repeated >= 64)
        throw std::runtime_error("repeated-token collapse");
      std::cout << id << "\n";
      emitted++;
      return !a_.ignore_eos && (id == 248044 || id == 248046);
    };
    bool stopped = output(current);
    while (!stopped && emitted < a_.max_new) {
      auto begin = std::chrono::steady_clock::now();
      auto result = round(current, a_.max_new - emitted, false);
      total += std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now() - begin)
                   .count();
      for (uint32_t i = 0; i < result.count; i++) {
        current = result.tokens[i];
        if (output(current)) {
          stopped = true;
          break;
        }
      }
    }
    std::cerr << "Decode "
              << (emitted > 1 && total ? 1000 * (emitted - 1) / total : 0)
              << " tok/s\n";
    if (a_.spec)
      std::cerr << "MTP metrics " << metrics_json() << "\n";
  }
};
} // namespace jr::v11
int main(int argc, char **argv) {
  try {
    // Text CLI and API share the existing Python tokenizer and chat template.
    // argv passes directly to exec: prompt contents are never interpreted by a
    // shell.
    bool frontend = false;
    for (int i = 1; i < argc; i++)
      if (std::string(argv[i]) == "--prompt" ||
          std::string(argv[i]) == "--prompt-file" ||
          std::string(argv[i]) == "--config" ||
          std::string(argv[i]) == "--help")
        frontend = true;
    if (frontend) {
      const char *python = std::getenv("JR_VK_PYTHON");
      if (!python)
        python = "python3";
      std::vector<std::string> args{
          python, JR_V11_FRONTEND, "--runtime",
          std::filesystem::absolute(argv[0]).string()};
      for (int i = 1; i < argc; i++)
        args.emplace_back(argv[i]);
      std::vector<char *> ptr;
      for (auto &x : args)
        ptr.push_back(x.data());
      ptr.push_back(nullptr);
      execvp(python, ptr.data());
      throw std::runtime_error("cannot start Python tokenizer frontend");
    }
    auto a = jr::v11::options(argc, argv);
    auto *old_stdout = std::cout.rdbuf(std::cerr.rdbuf());
    jr::v11::Runtime runtime(a);
    std::cout.rdbuf(old_stdout);
    if (a.worker)
      runtime.worker();
    else if (!a.ids.empty())
      runtime.ids();
    else
      throw std::runtime_error("use --prompt, --ids or --worker");
    runtime.export_profile();
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "V11 ERROR: " << e.what() << "\n";
    return 1;
  }
}
