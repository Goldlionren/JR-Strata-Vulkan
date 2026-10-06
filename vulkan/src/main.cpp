#include "jrstrata/vk/vk_context.hpp"
#include <vulkan/vulkan.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifndef JR_VK_VECTOR_SHADER
#error JR_VK_VECTOR_SHADER must be defined by CMake
#endif

namespace {
using jr::vk::vk_check;

struct Buffer {
    VkDevice device = VK_NULL_HANDLE;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    void* mapped = nullptr;
    bool coherent = false;

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
        coherent = o.coherent;
        o.device = VK_NULL_HANDLE;
        o.buffer = VK_NULL_HANDLE;
        o.memory = VK_NULL_HANDLE;
        o.size = 0;
        o.mapped = nullptr;
        o.coherent = false;
        return *this;
    }

    ~Buffer() { cleanup(); }

    void cleanup() {
        if (mapped && memory) vkUnmapMemory(device, memory);
        if (buffer) vkDestroyBuffer(device, buffer, nullptr);
        if (memory) vkFreeMemory(device, memory, nullptr);
        mapped = nullptr;
        buffer = VK_NULL_HANDLE;
        memory = VK_NULL_HANDLE;
    }
};

std::vector<uint32_t> read_spv(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot open SPIR-V: " + path);
    const auto bytes = f.tellg();
    if (bytes <= 0 || (static_cast<size_t>(bytes) % 4) != 0)
        throw std::runtime_error("invalid SPIR-V size");
    std::vector<uint32_t> code(static_cast<size_t>(bytes) / 4);
    f.seekg(0);
    f.read(reinterpret_cast<char*>(code.data()), bytes);
    if (!f) throw std::runtime_error("failed to read SPIR-V");
    return code;
}

Buffer make_host_buffer(const jr::vk::Context& ctx, VkDeviceSize size, VkBufferUsageFlags usage) {
    Buffer b;
    b.device = ctx.device();
    b.size = size;

    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = size;
    bci.usage = usage;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    vk_check(vkCreateBuffer(ctx.device(), &bci, nullptr, &b.buffer), "vkCreateBuffer");

    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(ctx.device(), b.buffer, &req);
    const uint32_t mt = ctx.find_memory_type(req.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    VkPhysicalDeviceMemoryProperties mp{};
    vkGetPhysicalDeviceMemoryProperties(ctx.physical_device(), &mp);
    b.coherent = (mp.memoryTypes[mt].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;

    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = mt;
    vk_check(vkAllocateMemory(ctx.device(), &mai, nullptr, &b.memory), "vkAllocateMemory");
    vk_check(vkBindBufferMemory(ctx.device(), b.buffer, b.memory, 0), "vkBindBufferMemory");
    vk_check(vkMapMemory(ctx.device(), b.memory, 0, size, 0, &b.mapped), "vkMapMemory");
    return b;
}

void flush_if_needed(const Buffer& b) {
    if (b.coherent) return;
    VkMappedMemoryRange r{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
    r.memory = b.memory;
    r.offset = 0;
    r.size = VK_WHOLE_SIZE;
    vk_check(vkFlushMappedMemoryRanges(b.device, 1, &r), "vkFlushMappedMemoryRanges");
}

void invalidate_if_needed(const Buffer& b) {
    if (b.coherent) return;
    VkMappedMemoryRange r{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
    r.memory = b.memory;
    r.offset = 0;
    r.size = VK_WHOLE_SIZE;
    vk_check(vkInvalidateMappedMemoryRanges(b.device, 1, &r), "vkInvalidateMappedMemoryRanges");
}

void print_caps(const jr::vk::DeviceCaps& c) {
    std::cout << "\nJR-VK selected device\n";
    std::cout << "  name               : " << c.device_name << "\n";
    std::cout << "  vendor:device      : " << std::hex << std::setfill('0')
              << std::setw(4) << c.vendor_id << ":" << std::setw(4) << c.device_id << std::dec << "\n";
    std::cout << "  Vulkan device API  : " << VK_VERSION_MAJOR(c.api_version) << "."
              << VK_VERSION_MINOR(c.api_version) << "." << VK_VERSION_PATCH(c.api_version) << "\n";
    std::cout << "  subgroup           : " << c.subgroup_size
              << " (min " << c.min_subgroup_size << ", max " << c.max_subgroup_size << ")\n";
    std::cout << "  bufferDeviceAddress: " << c.buffer_device_address << "\n";
    std::cout << "  timelineSemaphore  : " << c.timeline_semaphore << "\n";
    std::cout << "  shaderInt64        : " << c.shader_int64 << "\n";
    std::cout << "  shaderFloat16      : " << c.shader_float16 << "\n";
    std::cout << "  shaderInt8         : " << c.shader_int8 << "\n";
    std::cout << "  subgroup control   : " << c.subgroup_size_control << "\n";
    std::cout << "  external host mem  : " << c.ext_external_memory_host << "\n";
    std::cout << "  memory budget      : " << c.ext_memory_budget << "\n";
    std::cout << "  cooperative matrix : " << c.khr_cooperative_matrix << "\n";
    std::cout << "  host import align  : "
              << static_cast<unsigned long long>(c.min_imported_host_pointer_alignment) << " bytes\n";
    if (c.has_pci_bus_info) {
        std::cout << "  PCI                : " << c.pci_domain << ":" << c.pci_bus
                  << ":" << c.pci_device << "." << c.pci_function << "\n";
    }
}

} // namespace

int main() {
    try {
        jr::vk::Context ctx(jr::vk::selector_from_env());
        print_caps(ctx.caps());

        constexpr uint32_t N = 4096;
        constexpr VkDeviceSize bytes = N * sizeof(float);

        Buffer a = make_host_buffer(ctx, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        Buffer b = make_host_buffer(ctx, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        Buffer c = make_host_buffer(ctx, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

        auto* pa = static_cast<float*>(a.mapped);
        auto* pb = static_cast<float*>(b.mapped);
        auto* pc = static_cast<float*>(c.mapped);
        for (uint32_t i = 0; i < N; ++i) {
            pa[i] = static_cast<float>(i) * 0.25f;
            pb[i] = static_cast<float>(i % 97) * -0.5f;
            pc[i] = -9999.0f;
        }
        flush_if_needed(a);
        flush_if_needed(b);
        flush_if_needed(c);

        VkDescriptorSetLayoutBinding bindings[3]{};
        for (uint32_t i = 0; i < 3; ++i) {
            bindings[i].binding = i;
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }

        VkDescriptorSetLayoutCreateInfo dlci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        dlci.bindingCount = 3;
        dlci.pBindings = bindings;
        VkDescriptorSetLayout dlayout = VK_NULL_HANDLE;
        vk_check(vkCreateDescriptorSetLayout(ctx.device(), &dlci, nullptr, &dlayout), "vkCreateDescriptorSetLayout");

        VkPushConstantRange pcr{};
        pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcr.offset = 0;
        pcr.size = sizeof(uint32_t);

        VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        plci.setLayoutCount = 1;
        plci.pSetLayouts = &dlayout;
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges = &pcr;
        VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
        vk_check(vkCreatePipelineLayout(ctx.device(), &plci, nullptr, &pipeline_layout), "vkCreatePipelineLayout");

        const auto code = read_spv(JR_VK_VECTOR_SHADER);
        VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        smci.codeSize = code.size() * sizeof(uint32_t);
        smci.pCode = code.data();
        VkShaderModule shader = VK_NULL_HANDLE;
        vk_check(vkCreateShaderModule(ctx.device(), &smci, nullptr, &shader), "vkCreateShaderModule");

        VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = shader;
        stage.pName = "main";
        VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        cpi.stage = stage;
        cpi.layout = pipeline_layout;
        VkPipeline pipeline = VK_NULL_HANDLE;
        vk_check(vkCreateComputePipelines(ctx.device(), VK_NULL_HANDLE, 1, &cpi, nullptr, &pipeline),
                 "vkCreateComputePipelines");

        VkDescriptorPoolSize pool_size{};
        pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        pool_size.descriptorCount = 3;
        VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        dpci.maxSets = 1;
        dpci.poolSizeCount = 1;
        dpci.pPoolSizes = &pool_size;
        VkDescriptorPool pool = VK_NULL_HANDLE;
        vk_check(vkCreateDescriptorPool(ctx.device(), &dpci, nullptr, &pool), "vkCreateDescriptorPool");

        VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        dsai.descriptorPool = pool;
        dsai.descriptorSetCount = 1;
        dsai.pSetLayouts = &dlayout;
        VkDescriptorSet dset = VK_NULL_HANDLE;
        vk_check(vkAllocateDescriptorSets(ctx.device(), &dsai, &dset), "vkAllocateDescriptorSets");

        VkDescriptorBufferInfo infos[3]{};
        Buffer* bufs[3] = {&a, &b, &c};
        VkWriteDescriptorSet writes[3]{};
        for (uint32_t i = 0; i < 3; ++i) {
            infos[i].buffer = bufs[i]->buffer;
            infos[i].offset = 0;
            infos[i].range = bytes;
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = dset;
            writes[i].dstBinding = i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &infos[i];
        }
        vkUpdateDescriptorSets(ctx.device(), 3, writes, 0, nullptr);

        VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        cpci.queueFamilyIndex = ctx.queue_family();
        cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        VkCommandPool cmd_pool = VK_NULL_HANDLE;
        vk_check(vkCreateCommandPool(ctx.device(), &cpci, nullptr, &cmd_pool), "vkCreateCommandPool");

        VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cbai.commandPool = cmd_pool;
        cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbai.commandBufferCount = 1;
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        vk_check(vkAllocateCommandBuffers(ctx.device(), &cbai, &cmd), "vkAllocateCommandBuffers");

        VkCommandBufferBeginInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        cbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vk_check(vkBeginCommandBuffer(cmd, &cbi), "vkBeginCommandBuffer");
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, 1, &dset, 0, nullptr);
        vkCmdPushConstants(cmd, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t), &N);
        vkCmdDispatch(cmd, (N + 255u) / 256u, 1, 1);

        VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                             0, 1, &barrier, 0, nullptr, 0, nullptr);
        vk_check(vkEndCommandBuffer(cmd), "vkEndCommandBuffer");

        VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        VkFence fence = VK_NULL_HANDLE;
        vk_check(vkCreateFence(ctx.device(), &fci, nullptr, &fence), "vkCreateFence");
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd;
        vk_check(vkQueueSubmit(ctx.queue(), 1, &si, fence), "vkQueueSubmit");
        vk_check(vkWaitForFences(ctx.device(), 1, &fence, VK_TRUE, UINT64_MAX), "vkWaitForFences");
        invalidate_if_needed(c);

        uint32_t bad = 0;
        float max_abs_err = 0.0f;
        for (uint32_t i = 0; i < N; ++i) {
            const float expected = pa[i] + pb[i];
            const float err = std::fabs(pc[i] - expected);
            if (err > max_abs_err) max_abs_err = err;
            if (err > 1e-6f) {
                if (bad < 8) std::cerr << "mismatch i=" << i << " got=" << pc[i]
                                       << " expected=" << expected << "\n";
                ++bad;
            }
        }

        vkDestroyFence(ctx.device(), fence, nullptr);
        vkDestroyCommandPool(ctx.device(), cmd_pool, nullptr);
        vkDestroyDescriptorPool(ctx.device(), pool, nullptr);
        vkDestroyPipeline(ctx.device(), pipeline, nullptr);
        vkDestroyShaderModule(ctx.device(), shader, nullptr);
        vkDestroyPipelineLayout(ctx.device(), pipeline_layout, nullptr);
        vkDestroyDescriptorSetLayout(ctx.device(), dlayout, nullptr);

        std::cout << "\nJR-VK V0 vector-add\n";
        std::cout << "  elements     : " << N << "\n";
        std::cout << "  max abs error: " << max_abs_err << "\n";
        std::cout << "  mismatches   : " << bad << "\n";
        if (bad != 0) {
            std::cerr << "JR-VK V0: FAIL\n";
            return 2;
        }
        std::cout << "JR-VK V0: PASS\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "JR-VK V0 fatal: " << e.what() << "\n";
        return 1;
    }
}
