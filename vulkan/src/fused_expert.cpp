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

#ifndef JR_VK_FUSED_GEMV_SHADER
#error shader paths must be supplied by CMake
#endif

namespace {

using jr::vk::vk_check;
constexpr VkExternalMemoryHandleTypeFlagBits kHostHandle =
    VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;

struct Args {
    int gu_type = -1;
    int d_type = -1;
    std::string gate;
    std::string up;
    std::string down;
    std::string x;
    std::string out;
    uint32_t n_embd = 2560;
    uint32_t n_ff = 640;
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

    ~Pipeline() {
        if (pool) vkDestroyDescriptorPool(device, pool, nullptr);
        if (pipeline) vkDestroyPipeline(device, pipeline, nullptr);
        if (shader) vkDestroyShaderModule(device, shader, nullptr);
        if (layout) vkDestroyPipelineLayout(device, layout, nullptr);
        if (dlayout) vkDestroyDescriptorSetLayout(device, dlayout, nullptr);
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
        if (k == "--gu-type") a.gu_type = std::stoi(need("--gu-type"));
        else if (k == "--d-type") a.d_type = std::stoi(need("--d-type"));
        else if (k == "--gate") a.gate = need("--gate");
        else if (k == "--up") a.up = need("--up");
        else if (k == "--down") a.down = need("--down");
        else if (k == "--x") a.x = need("--x");
        else if (k == "--out") a.out = need("--out");
        else if (k == "--n-embd") a.n_embd = static_cast<uint32_t>(std::stoul(need("--n-embd")));
        else if (k == "--n-ff") a.n_ff = static_cast<uint32_t>(std::stoul(need("--n-ff")));
        else throw std::runtime_error("unknown argument: " + k);
    }

    if (a.gu_type < 0 || a.d_type < 0 || a.gate.empty() || a.up.empty() ||
        a.down.empty() || a.x.empty() || a.out.empty()) {
        throw std::runtime_error(
            "usage: jr-vk-full-expert --gu-type N --d-type N "
            "--gate gate.bin --up up.bin --down down.bin --x x.f32 --out y.f32");
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

size_t expected_role_bytes(int type_id, uint32_t rows, uint32_t cols) {
    return static_cast<size_t>(rows) * row_bytes_for(type_id, cols);
}

void fused_gemv(const jr::vk::Context& ctx,
                const Tables& tables,
                Buffer& qweights,
                Buffer& x,
                Buffer& y,
                uint32_t type_id,
                uint32_t rows,
                uint32_t cols) {
    struct Push {
        uint32_t type_id;
        uint32_t rows;
        uint32_t cols;
        uint32_t row_bytes;
    } push{
        type_id,
        rows,
        cols,
        static_cast<uint32_t>(row_bytes_for(type_id, cols))
    };

    std::vector<VkDescriptorBufferInfo> d = {
        info(qweights), info(x), info(y),
        info(tables.iq2xxs), info(tables.iq2xs), info(tables.iq2s),
        info(tables.iq3xxs), info(tables.iq3s),
        info(tables.signs), info(tables.iq4)
    };

    auto p = make_pipeline(ctx, JR_VK_FUSED_GEMV_SHADER, d, sizeof(push), 32);
    dispatch(ctx, p, rows, &push, sizeof(push));
}

void swiglu(const jr::vk::Context& ctx,
            Buffer& gate,
            Buffer& up,
            Buffer& h,
            uint32_t n) {
    struct Push { uint32_t n; } push{n};
    std::vector<VkDescriptorBufferInfo> d = {info(gate), info(up), info(h)};
    auto p = make_pipeline(ctx, JR_VK_EXPERT_SWIGLU_SHADER, d, sizeof(push), 32);
    dispatch(ctx, p, (n + 255u) / 256u, &push, sizeof(push));
}

} // namespace

int main(int argc, char** argv) {
    try {
        Args a = parse_args(argc, argv);
        jr::vk::Context ctx(jr::vk::selector_from_env());

        auto gate_bytes = read_bytes(a.gate);
        auto up_bytes = read_bytes(a.up);
        auto down_bytes = read_bytes(a.down);
        auto x = read_f32(a.x);

        if (x.size() != a.n_embd)
            throw std::runtime_error("input vector length does not match n_embd");

        const size_t gate_need =
            expected_role_bytes(a.gu_type, a.n_ff, a.n_embd);
        const size_t down_need =
            expected_role_bytes(a.d_type, a.n_embd, a.n_ff);

        if (gate_bytes.size() != gate_need)
            throw std::runtime_error(
                "gate byte count does not match gu_type/dimensions: got " +
                std::to_string(gate_bytes.size()) + ", expected " +
                std::to_string(gate_need));
        if (up_bytes.size() != gate_need)
            throw std::runtime_error("up byte count differs from expected gate size");
        if (down_bytes.size() != down_need)
            throw std::runtime_error(
                "down byte count does not match d_type/dimensions: got " +
                std::to_string(down_bytes.size()) + ", expected " +
                std::to_string(down_need));

        auto t0 = std::chrono::steady_clock::now();

        Tables tables = upload_tables(ctx);

        // The production-shaped memory path: the quantized expert itself stays
        // in imported host RAM. No full dequantized matrix is ever allocated.
        Buffer gate_q = make_imported_host(ctx, gate_bytes);
        Buffer up_q = make_imported_host(ctx, up_bytes);
        Buffer down_q = make_imported_host(ctx, down_bytes);

        Buffer x_dev = upload_device(ctx, x.data(), x.size() * sizeof(float));

        Buffer gate_v = make_buffer(
            ctx, a.n_ff * sizeof(float),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        Buffer up_v = make_buffer(
            ctx, a.n_ff * sizeof(float),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        Buffer h = make_buffer(
            ctx, a.n_ff * sizeof(float),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        Buffer y_dev = make_buffer(
            ctx, a.n_embd * sizeof(float),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        Buffer y_host = make_buffer(
            ctx, a.n_embd * sizeof(float),
            VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            true);

        auto t1 = std::chrono::steady_clock::now();

        fused_gemv(
            ctx, tables, gate_q, x_dev, gate_v,
            static_cast<uint32_t>(a.gu_type), a.n_ff, a.n_embd);
        fused_gemv(
            ctx, tables, up_q, x_dev, up_v,
            static_cast<uint32_t>(a.gu_type), a.n_ff, a.n_embd);
        swiglu(ctx, gate_v, up_v, h, a.n_ff);
        fused_gemv(
            ctx, tables, down_q, h, y_dev,
            static_cast<uint32_t>(a.d_type), a.n_embd, a.n_ff);

        copy_buffer(ctx, y_dev.buffer, y_host.buffer, a.n_embd * sizeof(float));
        auto t2 = std::chrono::steady_clock::now();

        write_f32(a.out, static_cast<float*>(y_host.mapped), a.n_embd);

        const double setup_ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count();
        const double run_ms =
            std::chrono::duration<double, std::milli>(t2 - t1).count();

        std::cout << "JR-VK fused full expert\n";
        std::cout << "  device  : " << ctx.caps().device_name << "\n";
        std::cout << "  GU type : " << a.gu_type << "\n";
        std::cout << "  D type  : " << a.d_type << "\n";
        std::cout << "  n_embd  : " << a.n_embd << "\n";
        std::cout << "  n_ff    : " << a.n_ff << "\n";
        std::cout << "  weights : quantized imported system RAM\n";
        std::cout << "  F32 matrix materialization: NO\n";
        std::cout << "  setup   : " << setup_ms << " ms\n";
        std::cout << "  execute : " << run_ms
                  << " ms (gate + up + SwiGLU + down + readback)\n";
        std::cout << "JR-VK fused full expert: DONE\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "JR-VK fused full expert fatal: " << e.what() << "\n";
        return 1;
    }
}
