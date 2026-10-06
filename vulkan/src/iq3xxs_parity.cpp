#include "jrstrata/vk/vk_context.hpp"
#include "strata/kernels/f16_bits.hpp"

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

#include <vulkan/vulkan.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifndef JR_VK_IQ3XXS_SHADER
#error JR_VK_IQ3XXS_SHADER must be supplied by CMake
#endif

namespace {

using jr::vk::vk_check;
using strata::kernels::f16_from_f32;
using strata::kernels::f32_from_f16;

constexpr VkExternalMemoryHandleTypeFlagBits kHostHandle =
    VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;

struct Buffer {
    VkDevice device = VK_NULL_HANDLE;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    void* mapped = nullptr;
    bool imported = false;
    bool owns_host = false;

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
        imported = o.imported;
        owns_host = o.owns_host;

        o.device = VK_NULL_HANDLE;
        o.buffer = VK_NULL_HANDLE;
        o.memory = VK_NULL_HANDLE;
        o.size = 0;
        o.mapped = nullptr;
        o.imported = false;
        o.owns_host = false;
        return *this;
    }

    ~Buffer() { cleanup(); }

    void cleanup() {
        if (mapped && memory && !imported) {
            vkUnmapMemory(device, memory);
        }
        if (buffer) vkDestroyBuffer(device, buffer, nullptr);
        if (memory) vkFreeMemory(device, memory, nullptr);
        if (owns_host && mapped) std::free(mapped);

        buffer = VK_NULL_HANDLE;
        memory = VK_NULL_HANDLE;
        mapped = nullptr;
        imported = false;
        owns_host = false;
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

uint32_t env_blocks() {
    const char* e = std::getenv("JR_VK_IQ3_BLOCKS");
    if (!e || !*e) return 4096;
    char* end = nullptr;
    unsigned long v = std::strtoul(e, &end, 10);
    if (!end || *end || v == 0 || v > (1u << 20)) {
        throw std::runtime_error("JR_VK_IQ3_BLOCKS must be 1..1048576");
    }
    return static_cast<uint32_t>(v);
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
        const auto flags = mp.memoryTypes[i].propertyFlags;
        if ((flags & required) != required) continue;

        if (fallback < 0) fallback = static_cast<int>(i);
        if ((flags & preferred) == preferred) return i;
    }

    if (fallback >= 0) return static_cast<uint32_t>(fallback);
    throw std::runtime_error("no compatible Vulkan memory type");
}

Buffer make_host_buffer(const jr::vk::Context& ctx,
                        VkDeviceSize size,
                        VkBufferUsageFlags usage) {
    Buffer b;
    b.device = ctx.device();
    b.size = size;

    VkBufferCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    ci.size = size;
    ci.usage = usage;
    ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    vk_check(vkCreateBuffer(ctx.device(), &ci, nullptr, &b.buffer),
             "vkCreateBuffer(host)");

    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(ctx.device(), b.buffer, &req);

    VkMemoryAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = pick_memory_type(
        ctx, req.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    vk_check(vkAllocateMemory(ctx.device(), &ai, nullptr, &b.memory),
             "vkAllocateMemory(host)");
    vk_check(vkBindBufferMemory(ctx.device(), b.buffer, b.memory, 0),
             "vkBindBufferMemory(host)");
    vk_check(vkMapMemory(ctx.device(), b.memory, 0, size, 0, &b.mapped),
             "vkMapMemory(host)");

    return b;
}

Buffer make_device_buffer(const jr::vk::Context& ctx,
                          VkDeviceSize size,
                          VkBufferUsageFlags usage) {
    Buffer b;
    b.device = ctx.device();
    b.size = size;

    VkBufferCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    ci.size = size;
    ci.usage = usage;
    ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    vk_check(vkCreateBuffer(ctx.device(), &ci, nullptr, &b.buffer),
             "vkCreateBuffer(device)");

    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(ctx.device(), b.buffer, &req);

    VkMemoryAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = pick_memory_type(
        ctx, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    vk_check(vkAllocateMemory(ctx.device(), &ai, nullptr, &b.memory),
             "vkAllocateMemory(device)");
    vk_check(vkBindBufferMemory(ctx.device(), b.buffer, b.memory, 0),
             "vkBindBufferMemory(device)");
    return b;
}

Buffer make_imported_host_buffer(const jr::vk::Context& ctx,
                                 VkDeviceSize requested_size) {
    auto get_props =
        reinterpret_cast<PFN_vkGetMemoryHostPointerPropertiesEXT>(
            vkGetDeviceProcAddr(
                ctx.device(), "vkGetMemoryHostPointerPropertiesEXT"));
    if (!get_props) {
        throw std::runtime_error(
            "vkGetMemoryHostPointerPropertiesEXT unavailable");
    }

    const VkDeviceSize align =
        std::max<VkDeviceSize>(
            ctx.caps().min_imported_host_pointer_alignment, 4096);
    const VkDeviceSize size =
        (requested_size + align - 1) & ~(align - 1);

    void* host = nullptr;
    const int rc = posix_memalign(
        &host, static_cast<size_t>(align), static_cast<size_t>(size));
    if (rc != 0 || !host) {
        throw std::runtime_error(
            std::string("posix_memalign failed: ") +
            std::strerror(rc ? rc : errno));
    }
    std::memset(host, 0, static_cast<size_t>(size));

    Buffer b;
    b.device = ctx.device();
    b.size = size;
    b.mapped = host;
    b.imported = true;
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
    if (!bits) {
        throw std::runtime_error(
            "imported host pointer has no compatible memory type");
    }

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
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    vk_check(vkAllocateMemory(ctx.device(), &ai, nullptr, &b.memory),
             "vkAllocateMemory(import)");
    vk_check(vkBindBufferMemory(ctx.device(), b.buffer, b.memory, 0),
             "vkBindBufferMemory(import)");

    return b;
}

void copy_buffer(const jr::vk::Context& ctx,
                 VkBuffer src,
                 VkBuffer dst,
                 VkDeviceSize size) {
    VkCommandPoolCreateInfo pci{};
    pci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pci.queueFamilyIndex = ctx.queue_family();

    VkCommandPool pool = VK_NULL_HANDLE;
    vk_check(vkCreateCommandPool(ctx.device(), &pci, nullptr, &pool),
             "vkCreateCommandPool(copy)");

    VkCommandBufferAllocateInfo cai{};
    cai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cai.commandPool = pool;
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    vk_check(vkAllocateCommandBuffers(ctx.device(), &cai, &cmd),
             "vkAllocateCommandBuffers(copy)");

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vk_check(vkBeginCommandBuffer(cmd, &bi),
             "vkBeginCommandBuffer(copy)");

    VkBufferCopy r{};
    r.size = size;
    vkCmdCopyBuffer(cmd, src, dst, 1, &r);

    vk_check(vkEndCommandBuffer(cmd),
             "vkEndCommandBuffer(copy)");

    VkFenceCreateInfo fi{};
    fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence fence = VK_NULL_HANDLE;
    vk_check(vkCreateFence(ctx.device(), &fi, nullptr, &fence),
             "vkCreateFence(copy)");

    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;

    vk_check(vkQueueSubmit(ctx.queue(), 1, &si, fence),
             "vkQueueSubmit(copy)");
    vk_check(vkWaitForFences(
                 ctx.device(), 1, &fence, VK_TRUE, UINT64_MAX),
             "vkWaitForFences(copy)");

    vkDestroyFence(ctx.device(), fence, nullptr);
    vkDestroyCommandPool(ctx.device(), pool, nullptr);
}

Pipeline make_pipeline(const jr::vk::Context& ctx,
                       VkBuffer src,
                       VkBuffer grid,
                       VkBuffer signs,
                       VkBuffer out,
                       VkDeviceSize src_size,
                       VkDeviceSize out_size) {
    Pipeline p;
    p.device = ctx.device();

    VkDescriptorSetLayoutBinding bindings[4]{};
    for (uint32_t i = 0; i < 4; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    VkDescriptorSetLayoutCreateInfo dlci{};
    dlci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dlci.bindingCount = 4;
    dlci.pBindings = bindings;

    vk_check(vkCreateDescriptorSetLayout(
                 ctx.device(), &dlci, nullptr, &p.dlayout),
             "vkCreateDescriptorSetLayout");

    VkPushConstantRange range{};
    range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    range.size = sizeof(uint32_t);

    VkPipelineLayoutCreateInfo plci{};
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &p.dlayout;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &range;

    vk_check(vkCreatePipelineLayout(
                 ctx.device(), &plci, nullptr, &p.layout),
             "vkCreatePipelineLayout");

    const auto code = read_spv(JR_VK_IQ3XXS_SHADER);
    VkShaderModuleCreateInfo smci{};
    smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = code.size() * sizeof(uint32_t);
    smci.pCode = code.data();

    vk_check(vkCreateShaderModule(
                 ctx.device(), &smci, nullptr, &p.shader),
             "vkCreateShaderModule");

    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo sg{};
    sg.sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO;
    sg.requiredSubgroupSize = 32;

    VkPipelineShaderStageCreateInfo stage{};
    stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage.pNext =
        (ctx.caps().subgroup_size_control &&
         ctx.caps().min_subgroup_size <= 32 &&
         ctx.caps().max_subgroup_size >= 32)
            ? &sg
            : nullptr;
    stage.flags = ctx.caps().compute_full_subgroups
        ? VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT
        : 0;
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = p.shader;
    stage.pName = "main";

    VkComputePipelineCreateInfo cpci{};
    cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.stage = stage;
    cpci.layout = p.layout;

    vk_check(vkCreateComputePipelines(
                 ctx.device(), VK_NULL_HANDLE,
                 1, &cpci, nullptr, &p.pipeline),
             "vkCreateComputePipelines");

    VkDescriptorPoolSize ps{};
    ps.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    ps.descriptorCount = 4;

    VkDescriptorPoolCreateInfo dpci{};
    dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpci.maxSets = 1;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes = &ps;

    vk_check(vkCreateDescriptorPool(
                 ctx.device(), &dpci, nullptr, &p.pool),
             "vkCreateDescriptorPool");

    VkDescriptorSetAllocateInfo dsai{};
    dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsai.descriptorPool = p.pool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &p.dlayout;

    vk_check(vkAllocateDescriptorSets(
                 ctx.device(), &dsai, &p.set),
             "vkAllocateDescriptorSets");

    VkDescriptorBufferInfo infos[4]{};
    infos[0].buffer = src;
    infos[0].range = src_size;
    infos[1].buffer = grid;
    infos[1].range = 256 * sizeof(uint32_t);
    infos[2].buffer = signs;
    infos[2].range = 128 * sizeof(uint32_t);
    infos[3].buffer = out;
    infos[3].range = out_size;

    VkWriteDescriptorSet writes[4]{};
    for (uint32_t i = 0; i < 4; ++i) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = p.set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &infos[i];
    }

    vkUpdateDescriptorSets(ctx.device(), 4, writes, 0, nullptr);
    return p;
}

void run_kernel(const jr::vk::Context& ctx,
                Pipeline& p,
                uint32_t blocks) {
    VkCommandPoolCreateInfo pci{};
    pci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pci.queueFamilyIndex = ctx.queue_family();

    VkCommandPool pool = VK_NULL_HANDLE;
    vk_check(vkCreateCommandPool(ctx.device(), &pci, nullptr, &pool),
             "vkCreateCommandPool(run)");

    VkCommandBufferAllocateInfo cai{};
    cai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cai.commandPool = pool;
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    vk_check(vkAllocateCommandBuffers(ctx.device(), &cai, &cmd),
             "vkAllocateCommandBuffers(run)");

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vk_check(vkBeginCommandBuffer(cmd, &bi),
             "vkBeginCommandBuffer(run)");

    vkCmdBindPipeline(
        cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p.pipeline);
    vkCmdBindDescriptorSets(
        cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        p.layout, 0, 1, &p.set, 0, nullptr);
    vkCmdPushConstants(
        cmd, p.layout, VK_SHADER_STAGE_COMPUTE_BIT,
        0, sizeof(uint32_t), &blocks);
    vkCmdDispatch(cmd, blocks, 1, 1);

    VkMemoryBarrier mb{};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;

    vkCmdPipelineBarrier(
        cmd,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_HOST_BIT,
        0,
        1, &mb,
        0, nullptr,
        0, nullptr);

    vk_check(vkEndCommandBuffer(cmd),
             "vkEndCommandBuffer(run)");

    VkFenceCreateInfo fi{};
    fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence fence = VK_NULL_HANDLE;

    vk_check(vkCreateFence(
                 ctx.device(), &fi, nullptr, &fence),
             "vkCreateFence(run)");

    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;

    vk_check(vkQueueSubmit(
                 ctx.queue(), 1, &si, fence),
             "vkQueueSubmit(run)");
    vk_check(vkWaitForFences(
                 ctx.device(), 1, &fence,
                 VK_TRUE, UINT64_MAX),
             "vkWaitForFences(run)");

    vkDestroyFence(ctx.device(), fence, nullptr);
    vkDestroyCommandPool(ctx.device(), pool, nullptr);
}

void cpu_decode_block(const block_iq3_xxs& b,
                      float* out) {
    const float d0 = f32_from_f16(b.d);

    for (int tid = 0; tid < 32; ++tid) {
        const int il = tid / 8;
        const int ib = tid % 8;

        const uint8_t* q3 = b.qs + 8 * ib;

        uint16_t gas0 = 0;
        uint16_t gas1 = 0;
        std::memcpy(&gas0, b.qs + 64 + 4 * ib + 0, 2);
        std::memcpy(&gas1, b.qs + 64 + 4 * ib + 2, 2);

        const uint32_t aux32 =
            static_cast<uint32_t>(gas0) |
            (static_cast<uint32_t>(gas1) << 16);

        const float d =
            d0 * (0.5f + static_cast<float>(aux32 >> 28)) * 0.5f;

        const uint8_t signs =
            ksigns_iq2xs[
                (aux32 >> (7 * il)) & 127];

        const uint32_t g0 =
            iq3xxs_grid[q3[2 * il + 0]];
        const uint32_t g1 =
            iq3xxs_grid[q3[2 * il + 1]];

        float* y = out + 32 * ib + 8 * il;

        for (int j = 0; j < 4; ++j) {
            const uint32_t v0 =
                (g0 >> (8 * j)) & 0xffu;
            const uint32_t v1 =
                (g1 >> (8 * j)) & 0xffu;

            y[j] =
                d * static_cast<float>(v0) *
                ((signs & (1u << j)) ? -1.0f : 1.0f);
            y[j + 4] =
                d * static_cast<float>(v1) *
                ((signs & (1u << (j + 4))) ? -1.0f : 1.0f);
        }
    }
}

void compare(const char* label,
             const std::vector<float>& ref,
             const float* got,
             size_t n) {
    size_t bad = 0;
    size_t bit_diff = 0;
    double max_abs = 0.0;
    double max_rel = 0.0;

    for (size_t i = 0; i < n; ++i) {
        const float a = ref[i];
        const float b = got[i];

        uint32_t ua = 0, ub = 0;
        std::memcpy(&ua, &a, 4);
        std::memcpy(&ub, &b, 4);
        if (ua != ub) ++bit_diff;

        const double ae =
            std::fabs(static_cast<double>(a) -
                      static_cast<double>(b));
        const double re =
            ae / std::max(1e-30, std::fabs(static_cast<double>(a)));

        max_abs = std::max(max_abs, ae);
        max_rel = std::max(max_rel, re);

        const double tol =
            1e-6 + 2e-6 * std::fabs(static_cast<double>(a));

        if (!std::isfinite(b) || ae > tol) {
            if (bad < 8) {
                std::cerr
                    << label
                    << " mismatch[" << i << "]"
                    << " ref=" << std::setprecision(9) << a
                    << " got=" << b
                    << " abs=" << ae
                    << "\n";
            }
            ++bad;
        }
    }

    std::cout
        << label
        << ": bit-diff " << bit_diff << "/" << n
        << ", tolerance failures " << bad
        << ", max abs " << std::scientific << max_abs
        << ", max rel " << max_rel
        << std::defaultfloat
        << "\n";

    if (bad) {
        throw std::runtime_error(
            std::string(label) + " parity failed");
    }
}

} // namespace

int main() {
    try {
        static_assert(sizeof(block_iq3_xxs) == 98,
                      "unexpected ggml block_iq3_xxs size");

        jr::vk::Context ctx(jr::vk::selector_from_env());
        const uint32_t nblocks = env_blocks();

        std::cout << "JR-VK V1.0 IQ3_XXS dequant parity\n";
        std::cout << "  device       : "
                  << ctx.caps().device_name << "\n";
        std::cout << "  vendor:device: 8086:e211\n";
        std::cout << "  blocks       : "
                  << nblocks << "\n";
        std::cout << "  block bytes  : "
                  << sizeof(block_iq3_xxs) << "\n";
        std::cout << "  values/block : 256\n";
        std::cout << "  subgroup     : 32\n\n";

        std::mt19937 rng(0x4a52564bu);
        std::uniform_int_distribution<int> byte_dist(0, 255);

        std::vector<block_iq3_xxs> blocks(nblocks);
        for (uint32_t b = 0; b < nblocks; ++b) {
            for (auto& q : blocks[b].qs) {
                q = static_cast<uint8_t>(byte_dist(rng));
            }

            // Finite, exact-ish scales covering both signs and powers-of-two.
            float d =
                std::ldexp(1.0f + 0.125f * static_cast<float>(b & 3u),
                           -8 + static_cast<int>(b % 7u));
            if (b & 1u) d = -d;
            blocks[b].d = f16_from_f32(d);
        }

        const size_t nvalues =
            static_cast<size_t>(nblocks) * 256;
        std::vector<float> reference(nvalues);

        for (uint32_t b = 0; b < nblocks; ++b) {
            cpu_decode_block(
                blocks[b],
                reference.data() + static_cast<size_t>(b) * 256);
        }

        const VkDeviceSize raw_bytes =
            static_cast<VkDeviceSize>(blocks.size()) *
            sizeof(block_iq3_xxs);
        const VkDeviceSize src_buffer_bytes =
            (raw_bytes + 3) & ~VkDeviceSize(3);
        const VkDeviceSize out_bytes =
            static_cast<VkDeviceSize>(nvalues) * sizeof(float);

        // Official ggml lookup tables, uploaded once.
        std::vector<uint32_t> signs32(128);
        for (size_t i = 0; i < signs32.size(); ++i) {
            signs32[i] = ksigns_iq2xs[i];
        }

        auto grid = make_host_buffer(
            ctx,
            256 * sizeof(uint32_t),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        auto signs = make_host_buffer(
            ctx,
            128 * sizeof(uint32_t),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        auto output = make_host_buffer(
            ctx,
            out_bytes,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

        std::memcpy(
            grid.mapped,
            iq3xxs_grid,
            256 * sizeof(uint32_t));
        std::memcpy(
            signs.mapped,
            signs32.data(),
            128 * sizeof(uint32_t));

        // Path A: imported system RAM source, the future host-expert tier.
        auto imported =
            make_imported_host_buffer(ctx, src_buffer_bytes);
        std::memcpy(
            imported.mapped,
            blocks.data(),
            static_cast<size_t>(raw_bytes));

        std::memset(output.mapped, 0, static_cast<size_t>(out_bytes));
        {
            auto p = make_pipeline(
                ctx,
                imported.buffer,
                grid.buffer,
                signs.buffer,
                output.buffer,
                src_buffer_bytes,
                out_bytes);
            run_kernel(ctx, p, nblocks);
        }

        compare(
            "imported-host",
            reference,
            static_cast<const float*>(output.mapped),
            nvalues);

        // Path B: identical bytes copied to device-local VRAM.
        auto device_src =
            make_device_buffer(
                ctx,
                src_buffer_bytes,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                VK_BUFFER_USAGE_TRANSFER_DST_BIT);

        copy_buffer(
            ctx,
            imported.buffer,
            device_src.buffer,
            src_buffer_bytes);

        std::memset(output.mapped, 0, static_cast<size_t>(out_bytes));
        {
            auto p = make_pipeline(
                ctx,
                device_src.buffer,
                grid.buffer,
                signs.buffer,
                output.buffer,
                src_buffer_bytes,
                out_bytes);
            run_kernel(ctx, p, nblocks);
        }

        compare(
            "device-local",
            reference,
            static_cast<const float*>(output.mapped),
            nvalues);

        std::cout << "\nJR-VK V1.0 IQ3_XXS: PASS\n";
        return 0;

    } catch (const std::exception& e) {
        std::cerr
            << "JR-VK V1.0 fatal: "
            << e.what() << "\n";
        return 1;
    }
}
