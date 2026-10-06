#include "jrstrata/vk/vk_context.hpp"

#include <vulkan/vulkan.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifndef JR_VK_READ_SCALAR_SHADER
#error shader paths must be supplied by CMake
#endif

namespace {

using jr::vk::vk_check;

constexpr VkExternalMemoryHandleTypeFlagBits kHostHandle =
    VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;

struct Buffer {
    VkDevice device = VK_NULL_HANDLE;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    void* mapped = nullptr;
    bool owns_host = false;
    void* imported_host = nullptr;

    Buffer() = default;
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    Buffer(Buffer&& o) noexcept { *this = std::move(o); }
    Buffer& operator=(Buffer&& o) noexcept {
        if (this == &o) return *this;
        cleanup();
        device = o.device;
        buffer = o.buffer;
        memory = o.memory;
        size = o.size;
        mapped = o.mapped;
        owns_host = o.owns_host;
        imported_host = o.imported_host;
        o.device = VK_NULL_HANDLE;
        o.buffer = VK_NULL_HANDLE;
        o.memory = VK_NULL_HANDLE;
        o.size = 0;
        o.mapped = nullptr;
        o.owns_host = false;
        o.imported_host = nullptr;
        return *this;
    }

    ~Buffer() { cleanup(); }

    void cleanup() {
        if (mapped && memory && !imported_host) vkUnmapMemory(device, memory);
        if (buffer) vkDestroyBuffer(device, buffer, nullptr);
        if (memory) vkFreeMemory(device, memory, nullptr);
        if (owns_host && imported_host) std::free(imported_host);
        buffer = VK_NULL_HANDLE;
        memory = VK_NULL_HANDLE;
        mapped = nullptr;
        imported_host = nullptr;
        owns_host = false;
    }
};

struct CommandEnv {
    VkDevice device = VK_NULL_HANDLE;
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    VkQueryPool queries = VK_NULL_HANDLE;

    ~CommandEnv() {
        if (queries) vkDestroyQueryPool(device, queries, nullptr);
        if (fence) vkDestroyFence(device, fence, nullptr);
        if (pool) vkDestroyCommandPool(device, pool, nullptr);
    }
};

struct Pipeline {
    VkDevice device = VK_NULL_HANDLE;
    VkDescriptorSetLayout dlayout = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkShaderModule shader = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkDescriptorSet set = VK_NULL_HANDLE;

    ~Pipeline() {
        if (pool) vkDestroyDescriptorPool(device, pool, nullptr);
        if (pipeline) vkDestroyPipeline(device, pipeline, nullptr);
        if (shader) vkDestroyShaderModule(device, shader, nullptr);
        if (layout) vkDestroyPipelineLayout(device, layout, nullptr);
        if (dlayout) vkDestroyDescriptorSetLayout(device, dlayout, nullptr);
    }
};

uint64_t env_mib(const char* name, uint64_t fallback) {
    const char* e = std::getenv(name);
    if (!e || !*e) return fallback;
    char* end = nullptr;
    unsigned long long v = std::strtoull(e, &end, 10);
    if (!end || *end || v == 0) {
        throw std::runtime_error(std::string(name) + " must be a positive integer");
    }
    return static_cast<uint64_t>(v);
}

std::vector<uint32_t> read_spv(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot open SPIR-V: " + path);
    auto n = f.tellg();
    if (n <= 0 || (static_cast<size_t>(n) % 4) != 0) {
        throw std::runtime_error("invalid SPIR-V size");
    }
    std::vector<uint32_t> code(static_cast<size_t>(n) / 4);
    f.seekg(0);
    f.read(reinterpret_cast<char*>(code.data()), n);
    if (!f) throw std::runtime_error("failed to read SPIR-V");
    return code;
}

uint32_t pick_memory_type(const jr::vk::Context& ctx,
                          uint32_t bits,
                          VkMemoryPropertyFlags required,
                          VkMemoryPropertyFlags preferred = 0) {
    VkPhysicalDeviceMemoryProperties mp{};
    vkGetPhysicalDeviceMemoryProperties(ctx.physical_device(), &mp);
    int fallback = -1;
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
        if (!(bits & (1u << i))) continue;
        auto flags = mp.memoryTypes[i].propertyFlags;
        if ((flags & required) != required) continue;
        if (fallback < 0) fallback = static_cast<int>(i);
        if ((flags & preferred) == preferred) return i;
    }
    if (fallback >= 0) return static_cast<uint32_t>(fallback);
    throw std::runtime_error("no compatible memory type");
}

Buffer make_device_buffer(const jr::vk::Context& ctx, VkDeviceSize size) {
    Buffer b;
    b.device = ctx.device();
    b.size = size;

    VkBufferCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    ci.size = size;
    ci.usage =
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
        VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    vk_check(vkCreateBuffer(ctx.device(), &ci, nullptr, &b.buffer), "vkCreateBuffer");

    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(ctx.device(), b.buffer, &req);

    VkMemoryAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = pick_memory_type(
        ctx, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    vk_check(vkAllocateMemory(ctx.device(), &ai, nullptr, &b.memory), "vkAllocateMemory");
    vk_check(vkBindBufferMemory(ctx.device(), b.buffer, b.memory, 0), "vkBindBufferMemory");
    return b;
}

Buffer make_readback_buffer(const jr::vk::Context& ctx, VkDeviceSize size) {
    Buffer b;
    b.device = ctx.device();
    b.size = size;

    VkBufferCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    ci.size = size;
    ci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    vk_check(vkCreateBuffer(ctx.device(), &ci, nullptr, &b.buffer), "vkCreateBuffer(readback)");

    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(ctx.device(), b.buffer, &req);

    VkMemoryAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = pick_memory_type(
        ctx, req.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    vk_check(vkAllocateMemory(ctx.device(), &ai, nullptr, &b.memory), "vkAllocateMemory(readback)");
    vk_check(vkBindBufferMemory(ctx.device(), b.buffer, b.memory, 0), "vkBindBufferMemory(readback)");
    vk_check(vkMapMemory(ctx.device(), b.memory, 0, size, 0, &b.mapped), "vkMapMemory(readback)");
    return b;
}

Buffer make_imported_host_buffer(const jr::vk::Context& ctx, VkDeviceSize requested) {
    auto get_props = reinterpret_cast<PFN_vkGetMemoryHostPointerPropertiesEXT>(
        vkGetDeviceProcAddr(ctx.device(), "vkGetMemoryHostPointerPropertiesEXT"));
    if (!get_props) throw std::runtime_error("vkGetMemoryHostPointerPropertiesEXT missing");

    const VkDeviceSize align =
        std::max<VkDeviceSize>(ctx.caps().min_imported_host_pointer_alignment, 4096);
    const VkDeviceSize size = (requested + align - 1) & ~(align - 1);

    void* host = nullptr;
    int rc = posix_memalign(&host, static_cast<size_t>(align), static_cast<size_t>(size));
    if (rc != 0 || !host) {
        throw std::runtime_error("posix_memalign failed");
    }

    Buffer b;
    b.device = ctx.device();
    b.size = size;
    b.mapped = host;
    b.imported_host = host;
    b.owns_host = true;

    VkExternalMemoryBufferCreateInfo ext{};
    ext.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
    ext.handleTypes = kHostHandle;

    VkBufferCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    ci.pNext = &ext;
    ci.size = size;
    ci.usage =
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    vk_check(vkCreateBuffer(ctx.device(), &ci, nullptr, &b.buffer),
             "vkCreateBuffer(import)");

    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(ctx.device(), b.buffer, &req);

    VkMemoryHostPointerPropertiesEXT hp{};
    hp.sType = VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT;
    vk_check(get_props(ctx.device(), kHostHandle, host, &hp),
             "vkGetMemoryHostPointerPropertiesEXT");

    const uint32_t bits = hp.memoryTypeBits & req.memoryTypeBits;
    if (!bits) throw std::runtime_error("no compatible imported-host memory type");

    VkImportMemoryHostPointerInfoEXT imp{};
    imp.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT;
    imp.handleType = kHostHandle;
    imp.pHostPointer = host;

    VkMemoryAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.pNext = &imp;
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = pick_memory_type(
        ctx, bits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    vk_check(vkAllocateMemory(ctx.device(), &ai, nullptr, &b.memory),
             "vkAllocateMemory(import)");
    vk_check(vkBindBufferMemory(ctx.device(), b.buffer, b.memory, 0),
             "vkBindBufferMemory(import)");
    return b;
}

CommandEnv make_command_env(const jr::vk::Context& ctx) {
    CommandEnv e;
    e.device = ctx.device();

    VkCommandPoolCreateInfo cp{};
    cp.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cp.queueFamilyIndex = ctx.queue_family();
    cp.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    vk_check(vkCreateCommandPool(ctx.device(), &cp, nullptr, &e.pool), "vkCreateCommandPool");

    VkCommandBufferAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool = e.pool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    vk_check(vkAllocateCommandBuffers(ctx.device(), &ai, &e.cmd), "vkAllocateCommandBuffers");

    VkFenceCreateInfo fi{};
    fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    vk_check(vkCreateFence(ctx.device(), &fi, nullptr, &e.fence), "vkCreateFence");

    VkQueryPoolCreateInfo qi{};
    qi.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    qi.queryType = VK_QUERY_TYPE_TIMESTAMP;
    qi.queryCount = 2;
    vk_check(vkCreateQueryPool(ctx.device(), &qi, nullptr, &e.queries), "vkCreateQueryPool");
    return e;
}

double run_timed(const jr::vk::Context& ctx,
                 CommandEnv& e,
                 const std::function<void(VkCommandBuffer)>& rec) {
    vk_check(vkResetFences(ctx.device(), 1, &e.fence), "vkResetFences");
    vk_check(vkResetCommandBuffer(e.cmd, 0), "vkResetCommandBuffer");

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vk_check(vkBeginCommandBuffer(e.cmd, &bi), "vkBeginCommandBuffer");

    vkCmdResetQueryPool(e.cmd, e.queries, 0, 2);
    vkCmdWriteTimestamp(e.cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, e.queries, 0);
    rec(e.cmd);
    vkCmdWriteTimestamp(e.cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, e.queries, 1);
    vk_check(vkEndCommandBuffer(e.cmd), "vkEndCommandBuffer");

    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &e.cmd;
    vk_check(vkQueueSubmit(ctx.queue(), 1, &si, e.fence), "vkQueueSubmit");
    vk_check(vkWaitForFences(ctx.device(), 1, &e.fence, VK_TRUE, UINT64_MAX),
             "vkWaitForFences");

    uint64_t ts[2]{};
    vk_check(vkGetQueryPoolResults(
        ctx.device(), e.queries, 0, 2, sizeof(ts), ts, sizeof(uint64_t),
        VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT),
        "vkGetQueryPoolResults");

    VkPhysicalDeviceProperties p{};
    vkGetPhysicalDeviceProperties(ctx.physical_device(), &p);
    return (double(ts[1] - ts[0]) * double(p.limits.timestampPeriod)) * 1e-9;
}

Pipeline make_pipeline(const jr::vk::Context& ctx,
                       const char* shader_path,
                       VkBuffer src,
                       VkBuffer dst,
                       VkDeviceSize size) {
    Pipeline p;
    p.device = ctx.device();

    VkDescriptorSetLayoutBinding b[2]{};
    for (uint32_t i = 0; i < 2; ++i) {
        b[i].binding = i;
        b[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        b[i].descriptorCount = 1;
        b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    VkDescriptorSetLayoutCreateInfo di{};
    di.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    di.bindingCount = 2;
    di.pBindings = b;
    vk_check(vkCreateDescriptorSetLayout(ctx.device(), &di, nullptr, &p.dlayout),
             "vkCreateDescriptorSetLayout");

    VkPushConstantRange range{};
    range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    range.size = sizeof(uint32_t);

    VkPipelineLayoutCreateInfo li{};
    li.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    li.setLayoutCount = 1;
    li.pSetLayouts = &p.dlayout;
    li.pushConstantRangeCount = 1;
    li.pPushConstantRanges = &range;
    vk_check(vkCreatePipelineLayout(ctx.device(), &li, nullptr, &p.layout),
             "vkCreatePipelineLayout");

    auto code = read_spv(shader_path);
    VkShaderModuleCreateInfo sm{};
    sm.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    sm.codeSize = code.size() * sizeof(uint32_t);
    sm.pCode = code.data();
    vk_check(vkCreateShaderModule(ctx.device(), &sm, nullptr, &p.shader),
             "vkCreateShaderModule");

    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo sg{};
    sg.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO;
    sg.requiredSubgroupSize = 32;

    VkPipelineShaderStageCreateInfo stage{};
    stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage.pNext = (ctx.caps().subgroup_size_control &&
                   ctx.caps().min_subgroup_size <= 32 &&
                   ctx.caps().max_subgroup_size >= 32) ? &sg : nullptr;
    stage.flags = ctx.caps().compute_full_subgroups
        ? VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT
        : 0;
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = p.shader;
    stage.pName = "main";

    VkComputePipelineCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    ci.stage = stage;
    ci.layout = p.layout;
    vk_check(vkCreateComputePipelines(
        ctx.device(), VK_NULL_HANDLE, 1, &ci, nullptr, &p.pipeline),
        "vkCreateComputePipelines");

    VkDescriptorPoolSize ps{};
    ps.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    ps.descriptorCount = 2;

    VkDescriptorPoolCreateInfo dpi{};
    dpi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpi.maxSets = 1;
    dpi.poolSizeCount = 1;
    dpi.pPoolSizes = &ps;
    vk_check(vkCreateDescriptorPool(ctx.device(), &dpi, nullptr, &p.pool),
             "vkCreateDescriptorPool");

    VkDescriptorSetAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ai.descriptorPool = p.pool;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &p.dlayout;
    vk_check(vkAllocateDescriptorSets(ctx.device(), &ai, &p.set),
             "vkAllocateDescriptorSets");

    VkDescriptorBufferInfo infos[2]{};
    infos[0].buffer = src; infos[0].range = size;
    infos[1].buffer = dst; infos[1].range = size;

    VkWriteDescriptorSet w[2]{};
    for (uint32_t i = 0; i < 2; ++i) {
        w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[i].dstSet = p.set;
        w[i].dstBinding = i;
        w[i].descriptorCount = 1;
        w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w[i].pBufferInfo = &infos[i];
    }
    vkUpdateDescriptorSets(ctx.device(), 2, w, 0, nullptr);
    return p;
}

double best_bandwidth(const jr::vk::Context& ctx,
                      CommandEnv& env,
                      Pipeline& p,
                      uint32_t units,
                      uint64_t source_bytes) {
    std::vector<double> rates;
    for (int it = 0; it < 5; ++it) {
        double s = run_timed(ctx, env, [&](VkCommandBuffer cb) {
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p.pipeline);
            vkCmdBindDescriptorSets(
                cb, VK_PIPELINE_BIND_POINT_COMPUTE, p.layout,
                0, 1, &p.set, 0, nullptr);
            vkCmdPushConstants(
                cb, p.layout, VK_SHADER_STAGE_COMPUTE_BIT,
                0, sizeof(uint32_t), &units);
            vkCmdDispatch(cb, (units + 255u) / 256u, 1, 1);
        });
        if (it) {
            rates.push_back(
                (double(source_bytes) / double(1ull << 30)) / s);
        }
    }
    return *std::max_element(rates.begin(), rates.end());
}

void upload_to_device(const jr::vk::Context& ctx,
                      CommandEnv& env,
                      VkBuffer src,
                      VkBuffer dst,
                      VkDeviceSize bytes) {
    (void)run_timed(ctx, env, [&](VkCommandBuffer cb) {
        VkBufferCopy r{};
        r.size = bytes;
        vkCmdCopyBuffer(cb, src, dst, 1, &r);
    });
}

} // namespace

int main() {
    try {
        jr::vk::Context ctx(jr::vk::selector_from_env());

        const uint64_t mib = env_mib("JR_VK_BENCH_MIB", 256);
        const uint64_t bytes = mib << 20;
        if ((bytes % 64) != 0) throw std::runtime_error("size must be divisible by 64 bytes");

        auto host = make_imported_host_buffer(ctx, bytes);
        auto dev_src = make_device_buffer(ctx, bytes);
        auto dev_dst = make_device_buffer(ctx, bytes);
        auto readback = make_readback_buffer(ctx, 4096 * sizeof(uint32_t));

        auto* h = static_cast<uint32_t*>(host.mapped);
        const uint64_t words = bytes / sizeof(uint32_t);
        for (uint64_t i = 0; i < words; ++i) {
            h[i] = 0xa5a50000u ^ static_cast<uint32_t>(i);
        }

        CommandEnv env = make_command_env(ctx);
        upload_to_device(ctx, env, host.buffer, dev_src.buffer, bytes);

        struct Case {
            const char* name;
            const char* shader;
            uint32_t bytes_per_unit;
        };

        const Case cases[] = {
            {"scalar-4B", JR_VK_READ_SCALAR_SHADER, 4},
            {"uvec4-16B", JR_VK_READ_VEC4_SHADER, 16},
            {"uvec4x2-32B", JR_VK_READ_U2_SHADER, 32},
            {"uvec4x4-64B", JR_VK_READ_U4_SHADER, 64},
        };

        std::cout << "JR-VK V0.4 vectorized host-read probe\n";
        std::cout << "  device : " << ctx.caps().device_name << "\n";
        std::cout << "  size   : " << mib << " MiB\n";
        std::cout << "  subgroup requested: 32\n\n";

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Case             Host-import read   Device-local read\n";
        std::cout << "-----------------------------------------------------\n";

        for (const auto& c : cases) {
            const uint32_t units = static_cast<uint32_t>(bytes / c.bytes_per_unit);

            auto hp = make_pipeline(ctx, c.shader, host.buffer, dev_dst.buffer, bytes);
            auto dp = make_pipeline(ctx, c.shader, dev_src.buffer, dev_dst.buffer, bytes);

            double host_bw = best_bandwidth(ctx, env, hp, units, bytes);
            double dev_bw = best_bandwidth(ctx, env, dp, units, bytes);

            std::cout << std::left << std::setw(16) << c.name
                      << std::right << std::setw(8) << host_bw << " GiB/s"
                      << std::setw(13) << dev_bw << " GiB/s\n";
        }

        // Correctness using the widest path.
        auto check_pipe = make_pipeline(
            ctx, JR_VK_READ_U4_SHADER, host.buffer, dev_dst.buffer, bytes);
        const uint32_t check_units = static_cast<uint32_t>(bytes / 64);
        (void)best_bandwidth(ctx, env, check_pipe, check_units, bytes);

        (void)run_timed(ctx, env, [&](VkCommandBuffer cb) {
            VkMemoryBarrier mb{};
            mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            vkCmdPipelineBarrier(
                cb,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                0, 1, &mb, 0, nullptr, 0, nullptr);

            VkBufferCopy r{};
            r.size = 4096 * sizeof(uint32_t);
            vkCmdCopyBuffer(cb, dev_dst.buffer, readback.buffer, 1, &r);
        });

        uint32_t bad = 0;
        auto* rb = static_cast<uint32_t*>(readback.mapped);
        for (uint32_t i = 0; i < 4096; ++i) {
            uint32_t expected = h[i] ^ 0x9e3779b9u;
            if (rb[i] != expected) ++bad;
        }

        std::cout << "\nV0.4 correctness: "
                  << (bad ? "FAIL" : "PASS")
                  << " (" << bad << " mismatches)\n";

        if (bad) return 2;
        std::cout << "JR-VK V0.4: PASS\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "JR-VK V0.4 fatal: " << e.what() << "\n";
        return 1;
    }
}
