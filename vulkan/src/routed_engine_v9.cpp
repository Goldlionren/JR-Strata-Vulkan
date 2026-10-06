#include "jrstrata/vk/vk_context.hpp"

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

#include <vulkan/vulkan.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifndef JR_VK_V8_T16_SHADER
#error V8 shader paths must be supplied by CMake
#endif
#ifndef JR_VK_V8_T17_SHADER
#error V8 shader paths must be supplied by CMake
#endif
#ifndef JR_VK_V8_T18_SHADER
#error V8 shader paths must be supplied by CMake
#endif
#ifndef JR_VK_V8_T20_SHADER
#error V8 shader paths must be supplied by CMake
#endif
#ifndef JR_VK_V8_T21_SHADER
#error V8 shader paths must be supplied by CMake
#endif
#ifndef JR_VK_V8_T22_SHADER
#error V8 shader paths must be supplied by CMake
#endif
#ifndef JR_VK_V8_T42_SHADER
#error V8 shader paths must be supplied by CMake
#endif

namespace {

using jr::vk::vk_check;
constexpr VkExternalMemoryHandleTypeFlagBits kHostHandle =
    VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;

struct Args {
    std::string bundle;
    std::string x;
    std::string out;
    std::string residency;
    int warmup = 0;
    int iters = 1;
};

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
        if (mapped && memory && !imported) vkUnmapMemory(device, memory);
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

    Pipeline() = default;
    Pipeline(const Pipeline&) = delete;
    Pipeline& operator=(const Pipeline&) = delete;

    Pipeline(Pipeline&& o) noexcept { *this = std::move(o); }

    Pipeline& operator=(Pipeline&& o) noexcept {
        if (this == &o) return *this;
        cleanup();

        device = o.device;
        dlayout = o.dlayout;
        layout = o.layout;
        shader = o.shader;
        pipeline = o.pipeline;
        pool = o.pool;
        set = o.set;

        o.device = VK_NULL_HANDLE;
        o.dlayout = VK_NULL_HANDLE;
        o.layout = VK_NULL_HANDLE;
        o.shader = VK_NULL_HANDLE;
        o.pipeline = VK_NULL_HANDLE;
        o.pool = VK_NULL_HANDLE;
        o.set = VK_NULL_HANDLE;
        return *this;
    }

    ~Pipeline() { cleanup(); }

    void cleanup() {
        if (pool) vkDestroyDescriptorPool(device, pool, nullptr);
        if (pipeline) vkDestroyPipeline(device, pipeline, nullptr);
        if (shader) vkDestroyShaderModule(device, shader, nullptr);
        if (layout) vkDestroyPipelineLayout(device, layout, nullptr);
        if (dlayout) vkDestroyDescriptorSetLayout(device, dlayout, nullptr);

        dlayout = VK_NULL_HANDLE;
        layout = VK_NULL_HANDLE;
        shader = VK_NULL_HANDLE;
        pipeline = VK_NULL_HANDLE;
        pool = VK_NULL_HANDLE;
        set = VK_NULL_HANDLE;
    }
};

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        auto need = [&](const char* what) -> std::string {
            if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + what);
            return argv[++i];
        };
        std::string k = argv[i];
        if (k == "--bundle") a.bundle = need("--bundle");
        else if (k == "--x") a.x = need("--x");
        else if (k == "--out") a.out = need("--out");
        else if (k == "--residency") a.residency = need("--residency");
        else if (k == "--warmup") a.warmup = std::stoi(need("--warmup"));
        else if (k == "--iters") a.iters = std::stoi(need("--iters"));
        else throw std::runtime_error("unknown argument: " + k);
    }
    if (a.bundle.empty() || a.x.empty() || a.out.empty() ||
        a.residency.empty() || a.warmup < 0 || a.iters < 1) {
        throw std::runtime_error(
            "usage: jr-vk-routed-engine-v9 --bundle route.jrvk --x x.f32 "
            "--out y.f32 --residency cache.jrvkc9 [--warmup N] [--iters N]");
    }
    return a;
}

std::vector<uint8_t> read_bytes(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot open " + path);
    auto n = f.tellg();
    if (n < 0) throw std::runtime_error("tellg failed for " + path);
    std::vector<uint8_t> out(static_cast<size_t>(n));
    f.seekg(0);
    f.read(reinterpret_cast<char*>(out.data()), n);
    if (!f) throw std::runtime_error("read failed for " + path);
    return out;
}

std::vector<float> read_f32(const std::string& path) {
    auto b = read_bytes(path);
    if (b.size() % sizeof(float)) throw std::runtime_error("not f32 file: " + path);
    std::vector<float> v(b.size() / sizeof(float));
    std::memcpy(v.data(), b.data(), b.size());
    return v;
}

void write_f32(const std::string& path, const float* p, size_t n) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot create " + path);
    f.write(reinterpret_cast<const char*>(p), static_cast<std::streamsize>(n * sizeof(float)));
    if (!f) throw std::runtime_error("write failed for " + path);
}

std::vector<uint32_t> read_spv(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot open SPIR-V " + path);
    auto n = f.tellg();
    if (n <= 0 || (static_cast<size_t>(n) % 4)) throw std::runtime_error("bad SPIR-V size");
    std::vector<uint32_t> out(static_cast<size_t>(n) / 4);
    f.seekg(0);
    f.read(reinterpret_cast<char*>(out.data()), n);
    if (!f) throw std::runtime_error("SPIR-V read failed");
    return out;
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
                   bool map = false) {
    Buffer b;
    b.device = ctx.device();
    b.size = size;

    VkBufferCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    ci.size = size;
    ci.usage = usage;
    ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    vk_check(vkCreateBuffer(ctx.device(), &ci, nullptr, &b.buffer), "vkCreateBuffer");

    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(ctx.device(), b.buffer, &req);

    VkMemoryAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = pick_memory_type(ctx, req.memoryTypeBits, required);

    vk_check(vkAllocateMemory(ctx.device(), &ai, nullptr, &b.memory), "vkAllocateMemory");
    vk_check(vkBindBufferMemory(ctx.device(), b.buffer, b.memory, 0), "vkBindBufferMemory");

    if (map) {
        vk_check(vkMapMemory(ctx.device(), b.memory, 0, size, 0, &b.mapped), "vkMapMemory");
    }
    return b;
}

Buffer make_imported_host(const jr::vk::Context& ctx,
                          const std::vector<uint8_t>& data) {
    auto get_props = reinterpret_cast<PFN_vkGetMemoryHostPointerPropertiesEXT>(
        vkGetDeviceProcAddr(ctx.device(), "vkGetMemoryHostPointerPropertiesEXT"));
    if (!get_props) throw std::runtime_error("vkGetMemoryHostPointerPropertiesEXT missing");

    const VkDeviceSize align =
        std::max<VkDeviceSize>(ctx.caps().min_imported_host_pointer_alignment, 4096);
    const VkDeviceSize size =
        (static_cast<VkDeviceSize>(data.size()) + align - 1) & ~(align - 1);

    void* host = nullptr;
    if (posix_memalign(&host, static_cast<size_t>(align), static_cast<size_t>(size)) != 0 || !host)
        throw std::runtime_error("posix_memalign failed");
    std::memset(host, 0, static_cast<size_t>(size));
    std::memcpy(host, data.data(), data.size());

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
    ci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    vk_check(vkCreateBuffer(ctx.device(), &ci, nullptr, &b.buffer), "vkCreateBuffer(import)");

    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(ctx.device(), b.buffer, &req);

    VkMemoryHostPointerPropertiesEXT hp{};
    hp.sType = VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT;
    vk_check(get_props(ctx.device(), kHostHandle, host, &hp),
             "vkGetMemoryHostPointerPropertiesEXT");

    uint32_t bits = hp.memoryTypeBits & req.memoryTypeBits;
    if (!bits) throw std::runtime_error("no imported-host memory type");

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

    vk_check(vkAllocateMemory(ctx.device(), &ai, nullptr, &b.memory), "vkAllocateMemory(import)");
    vk_check(vkBindBufferMemory(ctx.device(), b.buffer, b.memory, 0), "vkBindBufferMemory(import)");
    return b;
}

void copy_buffer(const jr::vk::Context& ctx, VkBuffer src, VkBuffer dst, VkDeviceSize size) {
    VkCommandPoolCreateInfo pci{};
    pci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pci.queueFamilyIndex = ctx.queue_family();

    VkCommandPool pool = VK_NULL_HANDLE;
    vk_check(vkCreateCommandPool(ctx.device(), &pci, nullptr, &pool), "vkCreateCommandPool(copy)");

    VkCommandBufferAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool = pool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    vk_check(vkAllocateCommandBuffers(ctx.device(), &ai, &cmd), "vkAllocateCommandBuffers(copy)");

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vk_check(vkBeginCommandBuffer(cmd, &bi), "vkBeginCommandBuffer(copy)");

    VkBufferCopy r{};
    r.size = size;
    vkCmdCopyBuffer(cmd, src, dst, 1, &r);
    vk_check(vkEndCommandBuffer(cmd), "vkEndCommandBuffer(copy)");

    VkFenceCreateInfo fi{};
    fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence fence = VK_NULL_HANDLE;
    vk_check(vkCreateFence(ctx.device(), &fi, nullptr, &fence), "vkCreateFence(copy)");

    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    vk_check(vkQueueSubmit(ctx.queue(), 1, &si, fence), "vkQueueSubmit(copy)");
    vk_check(vkWaitForFences(ctx.device(), 1, &fence, VK_TRUE, UINT64_MAX), "vkWaitForFences(copy)");

    vkDestroyFence(ctx.device(), fence, nullptr);
    vkDestroyCommandPool(ctx.device(), pool, nullptr);
}

Buffer upload_device(const jr::vk::Context& ctx,
                     const void* data,
                     VkDeviceSize size,
                     VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) {
    Buffer staging = make_buffer(
        ctx, size,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        true);
    std::memcpy(staging.mapped, data, static_cast<size_t>(size));

    Buffer dst = make_buffer(
        ctx, size,
        usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        false);

    copy_buffer(ctx, staging.buffer, dst.buffer, size);
    return dst;
}

Pipeline make_pipeline(const jr::vk::Context& ctx,
                       const char* spv,
                       const std::vector<VkDescriptorBufferInfo>& infos,
                       uint32_t push_size,
                       uint32_t required_subgroup = 32) {
    Pipeline p;
    p.device = ctx.device();

    std::vector<VkDescriptorSetLayoutBinding> bindings(infos.size());
    for (uint32_t i = 0; i < infos.size(); ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    VkDescriptorSetLayoutCreateInfo dl{};
    dl.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dl.bindingCount = static_cast<uint32_t>(bindings.size());
    dl.pBindings = bindings.data();
    vk_check(vkCreateDescriptorSetLayout(ctx.device(), &dl, nullptr, &p.dlayout),
             "vkCreateDescriptorSetLayout");

    VkPushConstantRange range{};
    range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    range.size = push_size;

    VkPipelineLayoutCreateInfo li{};
    li.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    li.setLayoutCount = 1;
    li.pSetLayouts = &p.dlayout;
    li.pushConstantRangeCount = push_size ? 1u : 0u;
    li.pPushConstantRanges = push_size ? &range : nullptr;
    vk_check(vkCreatePipelineLayout(ctx.device(), &li, nullptr, &p.layout),
             "vkCreatePipelineLayout");

    auto code = read_spv(spv);
    VkShaderModuleCreateInfo sm{};
    sm.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    sm.codeSize = code.size() * sizeof(uint32_t);
    sm.pCode = code.data();
    vk_check(vkCreateShaderModule(ctx.device(), &sm, nullptr, &p.shader), "vkCreateShaderModule");

    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo sg{};
    sg.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO;
    sg.requiredSubgroupSize = required_subgroup;

    VkPipelineShaderStageCreateInfo stage{};
    stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage.pNext =
        (ctx.caps().subgroup_size_control &&
         ctx.caps().min_subgroup_size <= required_subgroup &&
         ctx.caps().max_subgroup_size >= required_subgroup) ? &sg : nullptr;
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = p.shader;
    stage.pName = "main";

    VkComputePipelineCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    ci.stage = stage;
    ci.layout = p.layout;
    vk_check(vkCreateComputePipelines(ctx.device(), VK_NULL_HANDLE, 1, &ci, nullptr, &p.pipeline),
             "vkCreateComputePipelines");

    VkDescriptorPoolSize ps{};
    ps.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    ps.descriptorCount = static_cast<uint32_t>(infos.size());

    VkDescriptorPoolCreateInfo dpi{};
    dpi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpi.maxSets = 1;
    dpi.poolSizeCount = 1;
    dpi.pPoolSizes = &ps;
    vk_check(vkCreateDescriptorPool(ctx.device(), &dpi, nullptr, &p.pool), "vkCreateDescriptorPool");

    VkDescriptorSetAllocateInfo dsai{};
    dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsai.descriptorPool = p.pool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &p.dlayout;
    vk_check(vkAllocateDescriptorSets(ctx.device(), &dsai, &p.set), "vkAllocateDescriptorSets");

    std::vector<VkWriteDescriptorSet> wr(infos.size());
    for (uint32_t i = 0; i < infos.size(); ++i) {
        wr[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wr[i].dstSet = p.set;
        wr[i].dstBinding = i;
        wr[i].descriptorCount = 1;
        wr[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        wr[i].pBufferInfo = &infos[i];
    }
    vkUpdateDescriptorSets(ctx.device(), static_cast<uint32_t>(wr.size()), wr.data(), 0, nullptr);
    return p;
}

void dispatch(const jr::vk::Context& ctx,
              Pipeline& p,
              uint32_t gx,
              const void* push,
              uint32_t push_size) {
    VkCommandPoolCreateInfo pci{};
    pci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pci.queueFamilyIndex = ctx.queue_family();

    VkCommandPool pool = VK_NULL_HANDLE;
    vk_check(vkCreateCommandPool(ctx.device(), &pci, nullptr, &pool), "vkCreateCommandPool(dispatch)");

    VkCommandBufferAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool = pool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    vk_check(vkAllocateCommandBuffers(ctx.device(), &ai, &cmd), "vkAllocateCommandBuffers(dispatch)");

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vk_check(vkBeginCommandBuffer(cmd, &bi), "vkBeginCommandBuffer(dispatch)");

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            p.layout, 0, 1, &p.set, 0, nullptr);
    if (push_size) {
        vkCmdPushConstants(cmd, p.layout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, push_size, push);
    }
    vkCmdDispatch(cmd, gx, 1, 1);
    vk_check(vkEndCommandBuffer(cmd), "vkEndCommandBuffer(dispatch)");

    VkFenceCreateInfo fi{};
    fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence fence = VK_NULL_HANDLE;
    vk_check(vkCreateFence(ctx.device(), &fi, nullptr, &fence), "vkCreateFence(dispatch)");

    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    vk_check(vkQueueSubmit(ctx.queue(), 1, &si, fence), "vkQueueSubmit(dispatch)");
    vk_check(vkWaitForFences(ctx.device(), 1, &fence, VK_TRUE, UINT64_MAX),
             "vkWaitForFences(dispatch)");

    vkDestroyFence(ctx.device(), fence, nullptr);
    vkDestroyCommandPool(ctx.device(), pool, nullptr);
}

VkDescriptorBufferInfo info(const Buffer& b, VkDeviceSize range = VK_WHOLE_SIZE) {
    VkDescriptorBufferInfo i{};
    i.buffer = b.buffer;
    i.offset = 0;
    i.range = range == VK_WHOLE_SIZE ? b.size : range;
    return i;
}

struct Tables {
    Buffer iq2xxs;
    Buffer iq2xs;
    Buffer iq2s;
    Buffer iq3xxs;
    Buffer iq3s;
    Buffer signs;
    Buffer iq4;
};

Tables upload_tables(const jr::vk::Context& ctx) {
    Tables t;

    t.iq2xxs = upload_device(ctx, iq2xxs_grid, sizeof(iq2xxs_grid));
    t.iq2xs  = upload_device(ctx, iq2xs_grid, sizeof(iq2xs_grid));
    t.iq2s   = upload_device(ctx, iq2s_grid, sizeof(iq2s_grid));
    t.iq3xxs = upload_device(ctx, iq3xxs_grid, sizeof(iq3xxs_grid));
    t.iq3s   = upload_device(ctx, iq3s_grid, sizeof(iq3s_grid));

    std::vector<uint32_t> signs32(128);
    for (size_t i = 0; i < signs32.size(); ++i) signs32[i] = ksigns_iq2xs[i];
    t.signs = upload_device(ctx, signs32.data(), signs32.size() * sizeof(uint32_t));

    std::vector<int32_t> iq4v(16);
    for (size_t i = 0; i < iq4v.size(); ++i) iq4v[i] = kvalues_iq4nl[i];
    t.iq4 = upload_device(ctx, iq4v.data(), iq4v.size() * sizeof(int32_t));
    return t;
}




size_t row_bytes_for(int type_id, uint32_t cols) {
    switch (type_id) {
        case 16: return static_cast<size_t>(cols / 256u) * 66u;
        case 17: return static_cast<size_t>(cols / 256u) * 74u;
        case 18: return static_cast<size_t>(cols / 256u) * 98u;
        case 20: return static_cast<size_t>(cols / 32u)  * 18u;
        case 21: return static_cast<size_t>(cols / 256u) * 110u;
        case 22: return static_cast<size_t>(cols / 256u) * 82u;
        case 42: return static_cast<size_t>(cols / 64u)  * 18u;
        default: throw std::runtime_error("unsupported expert type " + std::to_string(type_id));
    }
}

size_t q8_bytes(uint32_t n) {
    if (n % 32u) throw std::runtime_error("q8_1 width must be divisible by 32");
    return static_cast<size_t>(n / 32u) * 36u;
}

struct ExpertRecord {
    int32_t id = -1;
    int32_t gu_type = -1;
    int32_t d_type = -1;
    uint32_t tier = 0;
    float weight = 0.0f;
    std::vector<uint8_t> gate, up, down;
};

struct RouteBundle {
    uint32_t layer = 0;
    uint32_t n_embd = 0;
    uint32_t n_ff = 0;
    std::vector<ExpertRecord> experts;
};

template <typename T>
T read_scalar(std::ifstream& f, const char* what) {
    T v{};
    f.read(reinterpret_cast<char*>(&v), sizeof(T));
    if (!f) throw std::runtime_error(std::string("bundle read failed: ") + what);
    return v;
}

std::vector<uint8_t> read_blob(std::ifstream& f, uint32_t n, const char* what) {
    std::vector<uint8_t> b(n);
    f.read(reinterpret_cast<char*>(b.data()), n);
    if (!f) throw std::runtime_error(std::string("bundle blob read failed: ") + what);
    return b;
}

RouteBundle read_bundle(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open bundle " + path);

    char magic[8]{};
    f.read(magic, 8);
    const char expected[8] = {'J','R','V','K','R','T','1','\0'};
    if (!f || std::memcmp(magic, expected, 8) != 0)
        throw std::runtime_error("bad routed bundle magic");

    const uint32_t version = read_scalar<uint32_t>(f, "version");
    if (version != 1) throw std::runtime_error("unsupported routed bundle version");

    RouteBundle b;
    b.layer = read_scalar<uint32_t>(f, "layer");
    const uint32_t k = read_scalar<uint32_t>(f, "k");
    b.n_embd = read_scalar<uint32_t>(f, "n_embd");
    b.n_ff = read_scalar<uint32_t>(f, "n_ff");
    if (k < 1 || k > 15) throw std::runtime_error("bundle k must be 1..15");

    b.experts.reserve(k);
    for (uint32_t i = 0; i < k; ++i) {
        ExpertRecord e;
        e.id = read_scalar<int32_t>(f, "expert id");
        e.gu_type = read_scalar<int32_t>(f, "gu type");
        e.d_type = read_scalar<int32_t>(f, "d type");
        e.tier = read_scalar<uint32_t>(f, "tier");
        e.weight = read_scalar<float>(f, "route weight");
        const uint32_t ng = read_scalar<uint32_t>(f, "gate bytes");
        const uint32_t nu = read_scalar<uint32_t>(f, "up bytes");
        const uint32_t nd = read_scalar<uint32_t>(f, "down bytes");
        e.gate = read_blob(f, ng, "gate");
        e.up = read_blob(f, nu, "up");
        e.down = read_blob(f, nd, "down");
        b.experts.push_back(std::move(e));
    }
    return b;
}


struct CachePlan {
    uint32_t n_layers = 0;
    uint32_t n_expert = 0;
    uint32_t n_resident = 0;
    uint64_t budget_bytes = 0;
    uint64_t used_bytes = 0;
    std::vector<int32_t> slot;  // -1 = imported RAM; >=0 = resident VRAM slot
};

CachePlan read_cache_plan(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open residency plan " + path);

    char magic[8]{};
    f.read(magic, 8);
    const char expected[8] = {'J','R','V','K','C','9','\0','\0'};
    if (!f || std::memcmp(magic, expected, 8) != 0)
        throw std::runtime_error("bad V9 residency plan magic");

    const uint32_t version = read_scalar<uint32_t>(f, "cache version");
    if (version != 1) throw std::runtime_error("unsupported V9 residency version");

    CachePlan p;
    p.n_layers = read_scalar<uint32_t>(f, "cache n_layers");
    p.n_expert = read_scalar<uint32_t>(f, "cache n_expert");
    p.n_resident = read_scalar<uint32_t>(f, "cache n_resident");
    (void) read_scalar<uint32_t>(f, "cache flags");
    p.budget_bytes = read_scalar<uint64_t>(f, "cache budget");
    p.used_bytes = read_scalar<uint64_t>(f, "cache used");

    if (p.n_layers == 0 || p.n_expert == 0)
        throw std::runtime_error("invalid V9 residency dimensions");

    const uint64_t cells = uint64_t(p.n_layers) * uint64_t(p.n_expert);
    if (cells > (1ull << 30))
        throw std::runtime_error("unreasonable V9 residency table");

    p.slot.resize(static_cast<size_t>(cells));
    f.read(reinterpret_cast<char*>(p.slot.data()),
           static_cast<std::streamsize>(p.slot.size() * sizeof(int32_t)));
    if (!f) throw std::runtime_error("truncated V9 residency table");
    return p;
}

uint32_t apply_cache_plan(RouteBundle& route, const CachePlan& plan) {
    if (route.layer >= plan.n_layers)
        throw std::runtime_error("route layer outside V9 residency plan");

    uint32_t hits = 0;
    for (auto& e : route.experts) {
        if (e.id < 0 || static_cast<uint32_t>(e.id) >= plan.n_expert)
            throw std::runtime_error("route expert outside V9 residency plan");
        const size_t at =
            static_cast<size_t>(route.layer) * plan.n_expert +
            static_cast<uint32_t>(e.id);
        const bool resident = plan.slot[at] >= 0;
        e.tier = resident ? 1u : 0u;
        hits += resident ? 1u : 0u;
    }
    return hits;
}

size_t align_up(size_t v, size_t a) {
    return (v + a - 1) & ~(a - 1);
}

struct ArenaBuild {
    std::vector<uint8_t> ram;
    std::vector<uint8_t> vram;
    std::vector<uint32_t> meta;  // 8 uints per expert
    std::vector<float> weights;
    size_t ram_weight_bytes = 0;
    size_t vram_weight_bytes = 0;
};

uint32_t append_blob(std::vector<uint8_t>& arena, const std::vector<uint8_t>& blob) {
    const size_t off = align_up(arena.size(), 4);
    if (off > UINT32_MAX || blob.size() > UINT32_MAX - off)
        throw std::runtime_error("route arena exceeds 32-bit segment offset");
    arena.resize(off, 0);
    const uint32_t result = static_cast<uint32_t>(off);
    arena.insert(arena.end(), blob.begin(), blob.end());
    return result;
}

ArenaBuild build_arenas(const RouteBundle& route) {
    ArenaBuild a;
    a.meta.resize(route.experts.size() * 8u);
    a.weights.reserve(route.experts.size());

    for (size_t r = 0; r < route.experts.size(); ++r) {
        const auto& e = route.experts[r];
        if (e.tier > 1) throw std::runtime_error("expert tier must be 0 or 1");

        const size_t gu_row = row_bytes_for(e.gu_type, route.n_embd);
        const size_t d_row = row_bytes_for(e.d_type, route.n_ff);
        const size_t gu_need = static_cast<size_t>(route.n_ff) * gu_row;
        const size_t d_need = static_cast<size_t>(route.n_embd) * d_row;
        if (e.gate.size() != gu_need || e.up.size() != gu_need || e.down.size() != d_need)
            throw std::runtime_error("expert blob size/type mismatch");

        auto& arena = e.tier == 0 ? a.ram : a.vram;
        const uint32_t go = append_blob(arena, e.gate);
        const uint32_t uo = append_blob(arena, e.up);
        const uint32_t dno = append_blob(arena, e.down);

        uint32_t* m = a.meta.data() + r * 8u;
        m[0] = e.tier;
        m[1] = static_cast<uint32_t>(e.gu_type);
        m[2] = static_cast<uint32_t>(e.d_type);
        m[3] = go;
        m[4] = uo;
        m[5] = dno;
        m[6] = static_cast<uint32_t>(gu_row);
        m[7] = static_cast<uint32_t>(d_row);

        a.weights.push_back(e.weight);
        const size_t bytes = e.gate.size() + e.up.size() + e.down.size();
        if (e.tier == 0) a.ram_weight_bytes += bytes;
        else a.vram_weight_bytes += bytes;
    }

    // Vulkan storage descriptors need a real buffer even when one tier is empty.
    if (a.ram.empty()) a.ram.resize(4, 0);
    if (a.vram.empty()) a.vram.resize(4, 0);
    return a;
}

void cmd_compute_barrier(VkCommandBuffer cmd) {
    VkMemoryBarrier mb{};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(
        cmd,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 1, &mb, 0, nullptr, 0, nullptr);
}

struct Recorded {
    VkDevice device = VK_NULL_HANDLE;
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    VkQueryPool query = VK_NULL_HANDLE;

    ~Recorded() {
        if (query) vkDestroyQueryPool(device, query, nullptr);
        if (fence) vkDestroyFence(device, fence, nullptr);
        if (pool) vkDestroyCommandPool(device, pool, nullptr);
    }
};

Recorded record_engine(
    const jr::vk::Context& ctx,
    Pipeline& quant,
    Pipeline& gu,
    Pipeline& swiglu,
    Pipeline& down,
    Pipeline& combine,
    uint32_t k,
    uint32_t n_embd,
    uint32_t n_ff) {

    Recorded r;
    r.device = ctx.device();

    VkCommandPoolCreateInfo pci{};
    pci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pci.queueFamilyIndex = ctx.queue_family();
    vk_check(vkCreateCommandPool(ctx.device(), &pci, nullptr, &r.pool),
             "vkCreateCommandPool(V7)");

    VkCommandBufferAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool = r.pool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    vk_check(vkAllocateCommandBuffers(ctx.device(), &ai, &r.cmd),
             "vkAllocateCommandBuffers(V7)");

    VkFenceCreateInfo fi{};
    fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    vk_check(vkCreateFence(ctx.device(), &fi, nullptr, &r.fence),
             "vkCreateFence(V7)");

    VkQueryPoolCreateInfo qi{};
    qi.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    qi.queryType = VK_QUERY_TYPE_TIMESTAMP;
    qi.queryCount = 2;
    vk_check(vkCreateQueryPool(ctx.device(), &qi, nullptr, &r.query),
             "vkCreateQueryPool(V7)");

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vk_check(vkBeginCommandBuffer(r.cmd, &bi), "vkBeginCommandBuffer(V7)");

    vkCmdResetQueryPool(r.cmd, r.query, 0, 2);
    vkCmdWriteTimestamp(r.cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, r.query, 0);

    // 1. Input F32 -> q8_1.
    struct QPush { uint32_t n; } qp{n_embd};
    vkCmdBindPipeline(r.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, quant.pipeline);
    vkCmdBindDescriptorSets(
        r.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        quant.layout, 0, 1, &quant.set, 0, nullptr);
    vkCmdPushConstants(
        r.cmd, quant.layout, VK_SHADER_STAGE_COMPUTE_BIT,
        0, sizeof(qp), &qp);
    vkCmdDispatch(r.cmd, n_embd / 32u, 1, 1);
    cmd_compute_barrier(r.cmd);

    // 2. All selected experts' gate + up in one dispatch.
    struct GPush {
        uint32_t k, n_embd, n_ff, role;
    } gp{k, n_embd, n_ff, 0u};
    vkCmdBindPipeline(r.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, gu.pipeline);
    vkCmdBindDescriptorSets(
        r.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        gu.layout, 0, 1, &gu.set, 0, nullptr);
    vkCmdPushConstants(
        r.cmd, gu.layout, VK_SHADER_STAGE_COMPUTE_BIT,
        0, sizeof(gp), &gp);
    vkCmdDispatch(r.cmd, (2u * n_ff + 3u) / 4u, k, 1);
    cmd_compute_barrier(r.cmd);

    // 3. All selected experts' SwiGLU -> q8_1 in one dispatch.
    struct SPush { uint32_t k, n_ff; } sp{k, n_ff};
    vkCmdBindPipeline(r.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, swiglu.pipeline);
    vkCmdBindDescriptorSets(
        r.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        swiglu.layout, 0, 1, &swiglu.set, 0, nullptr);
    vkCmdPushConstants(
        r.cmd, swiglu.layout, VK_SHADER_STAGE_COMPUTE_BIT,
        0, sizeof(sp), &sp);
    vkCmdDispatch(r.cmd, n_ff / 32u, k, 1);
    cmd_compute_barrier(r.cmd);

    // 4. All selected experts' down in one dispatch.
    gp.role = 1u;
    vkCmdBindPipeline(r.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, down.pipeline);
    vkCmdBindDescriptorSets(
        r.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        down.layout, 0, 1, &down.set, 0, nullptr);
    vkCmdPushConstants(
        r.cmd, down.layout, VK_SHADER_STAGE_COMPUTE_BIT,
        0, sizeof(gp), &gp);
    vkCmdDispatch(r.cmd, (n_embd + 3u) / 4u, k, 1);
    cmd_compute_barrier(r.cmd);

    // 5. Route-weighted combine in one dispatch.
    struct CPush { uint32_t k, n_embd; } cp{k, n_embd};
    vkCmdBindPipeline(r.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, combine.pipeline);
    vkCmdBindDescriptorSets(
        r.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        combine.layout, 0, 1, &combine.set, 0, nullptr);
    vkCmdPushConstants(
        r.cmd, combine.layout, VK_SHADER_STAGE_COMPUTE_BIT,
        0, sizeof(cp), &cp);
    vkCmdDispatch(r.cmd, (n_embd + 255u) / 256u, 1, 1);

    vkCmdWriteTimestamp(r.cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, r.query, 1);
    vk_check(vkEndCommandBuffer(r.cmd), "vkEndCommandBuffer(V7)");
    return r;
}

double submit_once(const jr::vk::Context& ctx, Recorded& r, double timestamp_period_ns) {
    vk_check(vkResetFences(ctx.device(), 1, &r.fence), "vkResetFences(V7)");

    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &r.cmd;
    vk_check(vkQueueSubmit(ctx.queue(), 1, &si, r.fence), "vkQueueSubmit(V7)");
    vk_check(vkWaitForFences(ctx.device(), 1, &r.fence, VK_TRUE, UINT64_MAX),
             "vkWaitForFences(V7)");

    uint64_t ts[2]{};
    vk_check(vkGetQueryPoolResults(
                 ctx.device(), r.query, 0, 2,
                 sizeof(ts), ts, sizeof(uint64_t),
                 VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT),
             "vkGetQueryPoolResults(V7)");

    return double(ts[1] - ts[0]) * timestamp_period_ns * 1e-6;
}



const char* shader_for_quant_type(int32_t type_id) {
    switch (type_id) {
        case 16: return JR_VK_V8_T16_SHADER;
        case 17: return JR_VK_V8_T17_SHADER;
        case 18: return JR_VK_V8_T18_SHADER;
        case 20: return JR_VK_V8_T20_SHADER;
        case 21: return JR_VK_V8_T21_SHADER;
        case 22: return JR_VK_V8_T22_SHADER;
        case 42: return JR_VK_V8_T42_SHADER;
        default:
            throw std::runtime_error(
                "V8 has no specialized shader for quant type " +
                std::to_string(type_id));
    }
}

void validate_uniform_layer_formats(const RouteBundle& route) {
    if (route.experts.empty())
        throw std::runtime_error("empty route bundle");

    const int32_t gu = route.experts.front().gu_type;
    const int32_t dn = route.experts.front().d_type;
    for (const auto& e : route.experts) {
        if (e.gu_type != gu || e.d_type != dn) {
            throw std::runtime_error(
                "V8 production kernel requires one gu_type and one d_type per layer");
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        Args a = parse_args(argc, argv);
        RouteBundle route = read_bundle(a.bundle);
        CachePlan cache = read_cache_plan(a.residency);
        const uint32_t cache_hits = apply_cache_plan(route, cache);
        validate_uniform_layer_formats(route);
        auto x = read_f32(a.x);

        const int32_t gu_type = route.experts.front().gu_type;
        const int32_t d_type = route.experts.front().d_type;
        const char* gu_shader = shader_for_quant_type(gu_type);
        const char* d_shader = shader_for_quant_type(d_type);

        const uint32_t k = static_cast<uint32_t>(route.experts.size());
        if (route.n_embd == 0 || route.n_ff == 0 ||
            route.n_embd % 32u || route.n_ff % 32u)
            throw std::runtime_error("invalid routed dimensions");
        if (x.size() != route.n_embd)
            throw std::runtime_error("input vector length does not match bundle");

        jr::vk::Context ctx(jr::vk::selector_from_env());
        if (!ctx.caps().shader_integer_dot_product ||
            !ctx.caps().intdot_4x8_packed_signed_accelerated) {
            throw std::runtime_error(
                "V7 requires accelerated signed packed 4x8 integer dot");
        }
        Tables tables = upload_tables(ctx);
        ArenaBuild arenas = build_arenas(route);

        Buffer ram_arena = make_imported_host(ctx, arenas.ram);
        Buffer vram_arena = upload_device(
            ctx, arenas.vram.data(), arenas.vram.size(),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        Buffer meta = upload_device(
            ctx, arenas.meta.data(),
            arenas.meta.size() * sizeof(uint32_t),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        Buffer route_weights = upload_device(
            ctx, arenas.weights.data(),
            arenas.weights.size() * sizeof(float),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

        Buffer x_dev = upload_device(
            ctx, x.data(), x.size() * sizeof(float),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        Buffer x_q8 = make_buffer(
            ctx, q8_bytes(route.n_embd),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        Buffer gu_out = make_buffer(
            ctx, static_cast<VkDeviceSize>(k) * 2u * route.n_ff * sizeof(float),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        Buffer hidden_q8 = make_buffer(
            ctx, static_cast<VkDeviceSize>(k) * q8_bytes(route.n_ff),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        Buffer parts = make_buffer(
            ctx, static_cast<VkDeviceSize>(k) * route.n_embd * sizeof(float),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        Buffer output = make_buffer(
            ctx, static_cast<VkDeviceSize>(route.n_embd) * sizeof(float),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        Buffer out_host = make_buffer(
            ctx, static_cast<VkDeviceSize>(route.n_embd) * sizeof(float),
            VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            true);

        // Persistent pipelines + persistent descriptor sets.
        std::vector<VkDescriptorBufferInfo> qdesc = {info(x_dev), info(x_q8)};
        Pipeline p_quant = make_pipeline(
            ctx, JR_VK_Q8_QUANT_SHADER, qdesc, sizeof(uint32_t), 32);

        auto grouped_desc = [&](Buffer& q8in, Buffer& out) {
            return std::vector<VkDescriptorBufferInfo>{
                info(ram_arena), info(vram_arena), info(q8in), info(out), info(meta),
                info(tables.iq2xxs), info(tables.iq2xs), info(tables.iq2s),
                info(tables.iq3xxs), info(tables.iq3s),
                info(tables.signs), info(tables.iq4)
            };
        };

        auto gu_desc = grouped_desc(x_q8, gu_out);
        Pipeline p_gu = make_pipeline(
            ctx, gu_shader, gu_desc, 16, 32);

        std::vector<VkDescriptorBufferInfo> sdesc = {info(gu_out), info(hidden_q8)};
        Pipeline p_swiglu = make_pipeline(
            ctx, JR_VK_V5_GROUPED_SWIGLU_SHADER, sdesc, 8, 32);

        auto down_desc = grouped_desc(hidden_q8, parts);
        Pipeline p_down = make_pipeline(
            ctx, d_shader, down_desc, 16, 32);

        std::vector<VkDescriptorBufferInfo> cdesc = {
            info(parts), info(route_weights), info(output)
        };
        Pipeline p_combine = make_pipeline(
            ctx, JR_VK_V5_COMBINE_SHADER, cdesc, 8, 32);

        Recorded recorded = record_engine(
            ctx, p_quant, p_gu, p_swiglu, p_down, p_combine,
            k, route.n_embd, route.n_ff);

        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(ctx.physical_device(), &props);
        const double timestamp_period_ns = props.limits.timestampPeriod;

        for (int i = 0; i < a.warmup; ++i)
            (void) submit_once(ctx, recorded, timestamp_period_ns);

        double total_ms = 0.0;
        double best_ms = 1e100;
        double worst_ms = 0.0;
        for (int i = 0; i < a.iters; ++i) {
            const double ms = submit_once(ctx, recorded, timestamp_period_ns);
            total_ms += ms;
            best_ms = std::min(best_ms, ms);
            worst_ms = std::max(worst_ms, ms);
        }
        const double avg_ms = total_ms / a.iters;

        copy_buffer(
            ctx, output.buffer, out_host.buffer,
            static_cast<VkDeviceSize>(route.n_embd) * sizeof(float));
        write_f32(
            a.out,
            static_cast<const float*>(out_host.mapped),
            route.n_embd);

        const size_t weight_bytes =
            arenas.ram_weight_bytes + arenas.vram_weight_bytes;
        const double effective_gib_s =
            avg_ms > 0.0
                ? (double(weight_bytes) / double(1ull << 30)) / (avg_ms / 1000.0)
                : 0.0;

        uint32_t nv = 0, nr = 0;
        for (const auto& e : route.experts) e.tier ? ++nv : ++nr;

        std::cout << "JR-Strata-Vulkan V9 Profile-Driven Cache Engine\n";
        std::cout << "  device         : " << ctx.caps().device_name << "\n";
        std::cout << "  shader int-dot : "
                  << (ctx.caps().shader_integer_dot_product ? "YES" : "NO") << "\n";
        std::cout << "  packed S8 accel: "
                  << (ctx.caps().intdot_4x8_packed_signed_accelerated ? "YES" : "NO") << "\n";
        std::cout << "  dot kernel     : OpSDot packed 4x8 (accelerated)\n";
        std::cout << "  row grouping   : 4 rows/workgroup, 8 lanes/row\n";
        std::cout << "  GU quant type  : " << gu_type << " (specialized)\n";
        std::cout << "  down quant type: " << d_type << " (specialized)\n";
        std::cout << "  layer          : " << route.layer << "\n";
        std::cout << "  selected       : " << k << "\n";
        std::cout << "  cache hits     : " << cache_hits << "/" << k << "\n";
        std::cout << "  cache misses   : " << (k - cache_hits) << "/" << k << "\n";
        std::cout << "  cache resident : " << cache.n_resident
                  << "/" << (uint64_t(cache.n_layers) * cache.n_expert) << " experts\n";
        std::cout << "  cache budget   : "
                  << (double(cache.budget_bytes) / double(1ull << 30)) << " GiB\n";
        std::cout << "  cache used     : "
                  << (double(cache.used_bytes) / double(1ull << 30)) << " GiB\n";
        std::cout << "  VRAM/RAM       : " << nv << "/" << nr << "\n";
        std::cout << "  pipelines      : persistent\n";
        std::cout << "  descriptors    : persistent\n";
        std::cout << "  command buffer : persistent, 5 grouped dispatches\n";
        std::cout << "  expert table   : two-segment arena + offset metadata\n";
        std::cout << "  activation     : q8_1 packed\n";
        std::cout << "  weight dot     : packed signed-int8 OpSDot + block reuse\n";
        std::cout << "  RAM weights    : "
                  << (double(arenas.ram_weight_bytes) / (1 << 20)) << " MiB\n";
        std::cout << "  VRAM weights   : "
                  << (double(arenas.vram_weight_bytes) / (1 << 20)) << " MiB\n";
        std::cout << "  warmup/iters   : " << a.warmup << "/" << a.iters << "\n";
        std::cout << "  GPU best       : " << best_ms << " ms\n";
        std::cout << "  GPU avg        : " << avg_ms << " ms\n";
        std::cout << "  GPU worst      : " << worst_ms << " ms\n";
        std::cout << "  effective routed weight throughput: "
                  << effective_gib_s << " GiB/s\n";
        std::cout << "  route          :";
        for (const auto& e : route.experts) {
            std::cout << " " << e.id << ":" << e.weight
                      << (e.tier ? "[V]" : "[R]");
        }
        std::cout << "\n";
        std::cout << "JR-VK V9 profile-cache engine: DONE\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "JR-VK V9 fatal: " << e.what() << "\n";
        return 1;
    }
}
