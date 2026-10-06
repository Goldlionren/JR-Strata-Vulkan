#include "jrstrata/vk/vk_context.hpp"

#include <vulkan/vulkan.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
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
#include <vector>

#ifndef JR_VK_HOST_COPY_SHADER
#error JR_VK_HOST_COPY_SHADER must be defined by CMake
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
    bool coherent = false;
    bool owns_host = false;
    void* imported_host = nullptr;

    Buffer() = default;
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    Buffer(Buffer&& other) noexcept {
        *this = std::move(other);
    }

    Buffer& operator=(Buffer&& other) noexcept {
        if (this == &other) return *this;
        cleanup();
        device = other.device;
        buffer = other.buffer;
        memory = other.memory;
        size = other.size;
        mapped = other.mapped;
        coherent = other.coherent;
        owns_host = other.owns_host;
        imported_host = other.imported_host;

        other.device = VK_NULL_HANDLE;
        other.buffer = VK_NULL_HANDLE;
        other.memory = VK_NULL_HANDLE;
        other.size = 0;
        other.mapped = nullptr;
        other.owns_host = false;
        other.imported_host = nullptr;
        return *this;
    }

    ~Buffer() { cleanup(); }

    void cleanup() {
        if (mapped && memory && !imported_host) {
            vkUnmapMemory(device, memory);
        }
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
    VkQueryPool query_pool = VK_NULL_HANDLE;

    ~CommandEnv() {
        if (query_pool) vkDestroyQueryPool(device, query_pool, nullptr);
        if (fence) vkDestroyFence(device, fence, nullptr);
        if (pool) vkDestroyCommandPool(device, pool, nullptr);
    }
};

uint64_t env_mib(const char* name, uint64_t fallback) {
    const char* e = std::getenv(name);
    if (!e || !*e) return fallback;
    char* end = nullptr;
    unsigned long long v = std::strtoull(e, &end, 10);
    if (!end || *end || v == 0) {
        throw std::runtime_error(std::string(name) + " must be a positive integer MiB value");
    }
    return static_cast<uint64_t>(v);
}

std::vector<uint32_t> read_spv(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot open SPIR-V: " + path);
    const auto bytes = f.tellg();
    if (bytes <= 0 || (static_cast<size_t>(bytes) % 4) != 0) {
        throw std::runtime_error("invalid SPIR-V size");
    }
    std::vector<uint32_t> code(static_cast<size_t>(bytes) / 4);
    f.seekg(0);
    f.read(reinterpret_cast<char*>(code.data()), bytes);
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
        auto f = mp.memoryTypes[i].propertyFlags;
        if ((f & required) != required) continue;
        if (fallback < 0) fallback = static_cast<int>(i);
        if ((f & preferred) == preferred) return i;
    }
    if (fallback >= 0) return static_cast<uint32_t>(fallback);
    throw std::runtime_error("no compatible Vulkan memory type");
}

Buffer make_buffer(const jr::vk::Context& ctx,
                   VkDeviceSize size,
                   VkBufferUsageFlags usage,
                   VkMemoryPropertyFlags required,
                   VkMemoryPropertyFlags preferred = 0) {
    Buffer b;
    b.device = ctx.device();
    b.size = size;

    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = size;
    bci.usage = usage;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    vk_check(vkCreateBuffer(ctx.device(), &bci, nullptr, &b.buffer), "vkCreateBuffer");

    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(ctx.device(), b.buffer, &req);

    uint32_t mt = pick_memory_type(ctx, req.memoryTypeBits, required, preferred);

    VkPhysicalDeviceMemoryProperties mp{};
    vkGetPhysicalDeviceMemoryProperties(ctx.physical_device(), &mp);
    b.coherent =
        (mp.memoryTypes[mt].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;

    VkMemoryAllocateInfo mai{};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = mt;

    vk_check(vkAllocateMemory(ctx.device(), &mai, nullptr, &b.memory), "vkAllocateMemory");
    vk_check(vkBindBufferMemory(ctx.device(), b.buffer, b.memory, 0), "vkBindBufferMemory");

    if (required & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
        vk_check(vkMapMemory(ctx.device(), b.memory, 0, size, 0, &b.mapped), "vkMapMemory");
    }

    return b;
}

Buffer make_imported_host_buffer(const jr::vk::Context& ctx,
                                 VkDeviceSize requested_size) {
    if (!ctx.caps().ext_external_memory_host) {
        throw std::runtime_error("VK_EXT_external_memory_host unavailable");
    }

    auto get_props =
        reinterpret_cast<PFN_vkGetMemoryHostPointerPropertiesEXT>(
            vkGetDeviceProcAddr(ctx.device(), "vkGetMemoryHostPointerPropertiesEXT"));
    if (!get_props) {
        throw std::runtime_error("vkGetMemoryHostPointerPropertiesEXT unavailable");
    }

    const VkDeviceSize align =
        std::max<VkDeviceSize>(ctx.caps().min_imported_host_pointer_alignment, 4096);
    const VkDeviceSize size = (requested_size + align - 1) & ~(align - 1);

    void* host = nullptr;
    const int rc = posix_memalign(&host, static_cast<size_t>(align), static_cast<size_t>(size));
    if (rc != 0 || !host) {
        throw std::runtime_error(
            "posix_memalign failed: " + std::string(std::strerror(rc ? rc : errno)));
    }

    Buffer b;
    b.device = ctx.device();
    b.size = size;
    b.owns_host = true;
    b.imported_host = host;
    b.mapped = host;

    VkExternalMemoryBufferCreateInfo ext_bci{};
    ext_bci.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
    ext_bci.handleTypes = kHostHandle;

    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.pNext = &ext_bci;
    bci.size = size;
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    vk_check(vkCreateBuffer(ctx.device(), &bci, nullptr, &b.buffer),
             "vkCreateBuffer(imported host)");

    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(ctx.device(), b.buffer, &req);

    VkMemoryHostPointerPropertiesEXT hp{};
    hp.sType = VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT;
    vk_check(get_props(ctx.device(), kHostHandle, host, &hp),
             "vkGetMemoryHostPointerPropertiesEXT");

    const uint32_t candidates = hp.memoryTypeBits & req.memoryTypeBits;
    if (!candidates) {
        throw std::runtime_error(
            "host pointer memory types do not intersect buffer memory types");
    }

    // V0.3 deliberately requires coherent host import so the benchmark path
    // has unambiguous host/device visibility semantics.
    const uint32_t mt = pick_memory_type(
        ctx,
        candidates,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    VkPhysicalDeviceMemoryProperties mp{};
    vkGetPhysicalDeviceMemoryProperties(ctx.physical_device(), &mp);
    b.coherent =
        (mp.memoryTypes[mt].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;

    VkImportMemoryHostPointerInfoEXT import{};
    import.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT;
    import.handleType = kHostHandle;
    import.pHostPointer = host;

    VkMemoryAllocateInfo mai{};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.pNext = &import;
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = mt;

    vk_check(vkAllocateMemory(ctx.device(), &mai, nullptr, &b.memory),
             "vkAllocateMemory(imported host)");
    vk_check(vkBindBufferMemory(ctx.device(), b.buffer, b.memory, 0),
             "vkBindBufferMemory(imported host)");

    return b;
}

void flush_if_needed(const Buffer& b) {
    if (!b.mapped || b.coherent || b.imported_host) return;
    VkMappedMemoryRange r{};
    r.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
    r.memory = b.memory;
    r.offset = 0;
    r.size = VK_WHOLE_SIZE;
    vk_check(vkFlushMappedMemoryRanges(b.device, 1, &r), "vkFlushMappedMemoryRanges");
}

void invalidate_if_needed(const Buffer& b) {
    if (!b.mapped || b.coherent || b.imported_host) return;
    VkMappedMemoryRange r{};
    r.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
    r.memory = b.memory;
    r.offset = 0;
    r.size = VK_WHOLE_SIZE;
    vk_check(vkInvalidateMappedMemoryRanges(b.device, 1, &r),
             "vkInvalidateMappedMemoryRanges");
}

CommandEnv make_command_env(const jr::vk::Context& ctx) {
    CommandEnv e;
    e.device = ctx.device();

    VkCommandPoolCreateInfo cp{};
    cp.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cp.queueFamilyIndex = ctx.queue_family();
    cp.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    vk_check(vkCreateCommandPool(ctx.device(), &cp, nullptr, &e.pool),
             "vkCreateCommandPool");

    VkCommandBufferAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool = e.pool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    vk_check(vkAllocateCommandBuffers(ctx.device(), &ai, &e.cmd),
             "vkAllocateCommandBuffers");

    VkFenceCreateInfo fi{};
    fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    vk_check(vkCreateFence(ctx.device(), &fi, nullptr, &e.fence),
             "vkCreateFence");

    VkQueryPoolCreateInfo qi{};
    qi.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    qi.queryType = VK_QUERY_TYPE_TIMESTAMP;
    qi.queryCount = 2;
    vk_check(vkCreateQueryPool(ctx.device(), &qi, nullptr, &e.query_pool),
             "vkCreateQueryPool");

    return e;
}

double submit_and_time(const jr::vk::Context& ctx,
                       CommandEnv& e,
                       const std::function<void(VkCommandBuffer)>& record) {
    vk_check(vkResetFences(ctx.device(), 1, &e.fence), "vkResetFences");
    vk_check(vkResetCommandBuffer(e.cmd, 0), "vkResetCommandBuffer");

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vk_check(vkBeginCommandBuffer(e.cmd, &bi), "vkBeginCommandBuffer");

    vkCmdResetQueryPool(e.cmd, e.query_pool, 0, 2);
    vkCmdWriteTimestamp(e.cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, e.query_pool, 0);

    record(e.cmd);

    vkCmdWriteTimestamp(e.cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, e.query_pool, 1);
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
                 ctx.device(), e.query_pool, 0, 2,
                 sizeof(ts), ts, sizeof(uint64_t),
                 VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT),
             "vkGetQueryPoolResults");

    VkPhysicalDeviceProperties p{};
    vkGetPhysicalDeviceProperties(ctx.physical_device(), &p);
    const double ns = static_cast<double>(ts[1] - ts[0]) *
                      static_cast<double>(p.limits.timestampPeriod);
    return ns * 1e-9;
}

double gib_per_s(uint64_t bytes, double seconds) {
    return (static_cast<double>(bytes) / static_cast<double>(1ull << 30)) / seconds;
}

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

Pipeline make_copy_pipeline(const jr::vk::Context& ctx,
                            VkBuffer src,
                            VkBuffer dst,
                            VkDeviceSize size) {
    Pipeline p;
    p.device = ctx.device();

    VkDescriptorSetLayoutBinding bindings[2]{};
    for (uint32_t i = 0; i < 2; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    VkDescriptorSetLayoutCreateInfo dli{};
    dli.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dli.bindingCount = 2;
    dli.pBindings = bindings;
    vk_check(vkCreateDescriptorSetLayout(ctx.device(), &dli, nullptr, &p.dlayout),
             "vkCreateDescriptorSetLayout");

    VkPushConstantRange range{};
    range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    range.offset = 0;
    range.size = sizeof(uint32_t);

    VkPipelineLayoutCreateInfo pli{};
    pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &p.dlayout;
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges = &range;
    vk_check(vkCreatePipelineLayout(ctx.device(), &pli, nullptr, &p.layout),
             "vkCreatePipelineLayout");

    auto code = read_spv(JR_VK_HOST_COPY_SHADER);
    VkShaderModuleCreateInfo sm{};
    sm.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    sm.codeSize = code.size() * sizeof(uint32_t);
    sm.pCode = code.data();
    vk_check(vkCreateShaderModule(ctx.device(), &sm, nullptr, &p.shader),
             "vkCreateShaderModule");

    VkPipelineShaderStageCreateInfo stage{};
    stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
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
    infos[0].buffer = src;
    infos[0].offset = 0;
    infos[0].range = size;
    infos[1].buffer = dst;
    infos[1].offset = 0;
    infos[1].range = size;

    VkWriteDescriptorSet wr[2]{};
    for (uint32_t i = 0; i < 2; ++i) {
        wr[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wr[i].dstSet = p.set;
        wr[i].dstBinding = i;
        wr[i].descriptorCount = 1;
        wr[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        wr[i].pBufferInfo = &infos[i];
    }
    vkUpdateDescriptorSets(ctx.device(), 2, wr, 0, nullptr);

    return p;
}

} // namespace

int main() {
    try {
        jr::vk::Context ctx(jr::vk::selector_from_env());

        const uint64_t mib = env_mib("JR_VK_BENCH_MIB", 256);
        const uint64_t bytes = mib << 20;
        const uint32_t words = static_cast<uint32_t>(bytes / sizeof(uint32_t));

        if (bytes > 0xffffffffull * sizeof(uint32_t)) {
            throw std::runtime_error("benchmark size is too large for V0.3 shader indexing");
        }

        std::cout << "JR-VK memory probe\n";
        std::cout << "  device     : " << ctx.caps().device_name << "\n";
        std::cout << "  vendor:id  : 8086:e211\n";
        std::cout << "  size       : " << mib << " MiB\n";
        std::cout << "  host align : "
                  << static_cast<unsigned long long>(
                         ctx.caps().min_imported_host_pointer_alignment)
                  << " bytes\n\n";

        auto staging = make_buffer(
            ctx, bytes,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        auto readback = make_buffer(
            ctx, bytes,
            VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        auto device_buf = make_buffer(
            ctx, bytes,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
            VK_BUFFER_USAGE_TRANSFER_DST_BIT |
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        auto* s = static_cast<uint32_t*>(staging.mapped);
        for (uint32_t i = 0; i < words; ++i) s[i] = i * 2654435761u;
        flush_if_needed(staging);

        CommandEnv cmd = make_command_env(ctx);

        std::vector<double> h2d;
        std::vector<double> d2h;

        for (int it = 0; it < 4; ++it) {
            double sec = submit_and_time(ctx, cmd, [&](VkCommandBuffer cb) {
                VkBufferCopy r{};
                r.size = bytes;
                vkCmdCopyBuffer(cb, staging.buffer, device_buf.buffer, 1, &r);
            });
            if (it) h2d.push_back(gib_per_s(bytes, sec));
        }

        for (int it = 0; it < 4; ++it) {
            double sec = submit_and_time(ctx, cmd, [&](VkCommandBuffer cb) {
                VkBufferCopy r{};
                r.size = bytes;
                vkCmdCopyBuffer(cb, device_buf.buffer, readback.buffer, 1, &r);
            });
            if (it) d2h.push_back(gib_per_s(bytes, sec));
        }
        invalidate_if_needed(readback);

        auto imported = make_imported_host_buffer(ctx, bytes);
        auto* h = static_cast<uint32_t*>(imported.mapped);
        for (uint32_t i = 0; i < words; ++i) h[i] = 0xa5a50000u ^ i;

        Pipeline pipe = make_copy_pipeline(ctx, imported.buffer, device_buf.buffer, bytes);

        std::vector<double> direct;
        for (int it = 0; it < 4; ++it) {
            double sec = submit_and_time(ctx, cmd, [&](VkCommandBuffer cb) {
                vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.pipeline);
                vkCmdBindDescriptorSets(
                    cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                    pipe.layout, 0, 1, &pipe.set, 0, nullptr);
                vkCmdPushConstants(
                    cb, pipe.layout, VK_SHADER_STAGE_COMPUTE_BIT,
                    0, sizeof(uint32_t), &words);
                vkCmdDispatch(cb, (words + 255u) / 256u, 1, 1);
            });
            if (it) direct.push_back(gib_per_s(bytes, sec));
        }

        // Copy a small prefix back for correctness validation.
        constexpr VkDeviceSize check_bytes = 4096 * sizeof(uint32_t);
        double ignored = submit_and_time(ctx, cmd, [&](VkCommandBuffer cb) {
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
            r.size = check_bytes;
            vkCmdCopyBuffer(cb, device_buf.buffer, readback.buffer, 1, &r);
        });
        (void)ignored;

        invalidate_if_needed(readback);
        auto* rb = static_cast<uint32_t*>(readback.mapped);
        uint32_t bad = 0;
        for (uint32_t i = 0; i < 4096; ++i) {
            uint32_t expected = h[i] ^ 0x9e3779b9u;
            if (rb[i] != expected) {
                if (bad < 8) {
                    std::cerr << "mismatch i=" << i
                              << " got=0x" << std::hex << rb[i]
                              << " expected=0x" << expected
                              << std::dec << "\n";
                }
                ++bad;
            }
        }

        auto best = [](const std::vector<double>& v) {
            return *std::max_element(v.begin(), v.end());
        };
        auto avg = [](const std::vector<double>& v) {
            return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
        };

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "V0.2 vkCmdCopyBuffer H2D : best "
                  << best(h2d) << " GiB/s, avg " << avg(h2d) << " GiB/s\n";
        std::cout << "V0.2 vkCmdCopyBuffer D2H : best "
                  << best(d2h) << " GiB/s, avg " << avg(d2h) << " GiB/s\n";
        std::cout << "V0.3 imported host read  : best "
                  << best(direct) << " GiB/s, avg " << avg(direct) << " GiB/s\n";
        std::cout << "V0.3 correctness         : "
                  << (bad ? "FAIL" : "PASS")
                  << " (" << bad << " mismatches)\n";

        if (bad) return 2;

        std::cout << "JR-VK V0.3: PASS\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "JR-VK memprobe fatal: " << e.what() << "\n";
        return 1;
    }
}
