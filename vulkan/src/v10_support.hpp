// V10 uses the V9 allocation/import and V8 shader contracts verbatim.
#pragma once
#include "jrstrata/vk/vk_context.hpp"
#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
namespace jr::v10 {
using jr::vk::vk_check;
constexpr VkExternalMemoryHandleTypeFlagBits kHostHandle =
    VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
struct Buffer {
  VkDevice device = VK_NULL_HANDLE;
  VkBuffer buffer = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  VkDeviceSize size = 0;
  void *mapped = nullptr;
  bool imported = false;
  bool owns_host = false;

  Buffer() = default;
  Buffer(const Buffer &) = delete;
  Buffer &operator=(const Buffer &) = delete;

  Buffer(Buffer &&o) noexcept { *this = std::move(o); }
  Buffer &operator=(Buffer &&o) noexcept {
    if (this == &o)
      return *this;
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
    if (mapped && memory && !imported)
      vkUnmapMemory(device, memory);
    if (buffer)
      vkDestroyBuffer(device, buffer, nullptr);
    if (memory)
      vkFreeMemory(device, memory, nullptr);
    if (owns_host && mapped)
      std::free(mapped);
    buffer = VK_NULL_HANDLE;
    memory = VK_NULL_HANDLE;
    mapped = nullptr;
    imported = false;
    owns_host = false;
  }
};

// Compiled programs are shared across layers; only descriptor sets differ.
struct PipelineCore {
  VkDevice device = VK_NULL_HANDLE;
  VkDescriptorSetLayout dlayout = VK_NULL_HANDLE;
  VkPipelineLayout layout = VK_NULL_HANDLE;
  VkShaderModule shader = VK_NULL_HANDLE;
  VkPipeline pipeline = VK_NULL_HANDLE;
  ~PipelineCore() {
    if (pipeline)
      vkDestroyPipeline(device, pipeline, nullptr);
    if (shader)
      vkDestroyShaderModule(device, shader, nullptr);
    if (layout)
      vkDestroyPipelineLayout(device, layout, nullptr);
    if (dlayout)
      vkDestroyDescriptorSetLayout(device, dlayout, nullptr);
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
  std::shared_ptr<PipelineCore> core;

  Pipeline() = default;
  Pipeline(const Pipeline &) = delete;
  Pipeline &operator=(const Pipeline &) = delete;

  Pipeline(Pipeline &&o) noexcept { *this = std::move(o); }

  Pipeline &operator=(Pipeline &&o) noexcept {
    if (this == &o)
      return *this;
    cleanup();

    core = std::move(o.core);
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
    if (pool)
      vkDestroyDescriptorPool(device, pool, nullptr);
    core.reset();

    dlayout = VK_NULL_HANDLE;
    layout = VK_NULL_HANDLE;
    shader = VK_NULL_HANDLE;
    pipeline = VK_NULL_HANDLE;
    pool = VK_NULL_HANDLE;
    set = VK_NULL_HANDLE;
  }
};

std::vector<uint8_t> read_bytes(const std::string &path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f)
    throw std::runtime_error("cannot open " + path);
  auto n = f.tellg();
  if (n < 0)
    throw std::runtime_error("tellg failed for " + path);
  std::vector<uint8_t> out(static_cast<size_t>(n));
  f.seekg(0);
  f.read(reinterpret_cast<char *>(out.data()), n);
  if (!f)
    throw std::runtime_error("read failed for " + path);
  return out;
}

std::vector<float> read_f32(const std::string &path) {
  auto b = read_bytes(path);
  if (b.size() % sizeof(float))
    throw std::runtime_error("not f32 file: " + path);
  std::vector<float> v(b.size() / sizeof(float));
  std::memcpy(v.data(), b.data(), b.size());
  return v;
}

void write_f32(const std::string &path, const float *p, size_t n) {
  std::ofstream f(path, std::ios::binary);
  if (!f)
    throw std::runtime_error("cannot create " + path);
  f.write(reinterpret_cast<const char *>(p),
          static_cast<std::streamsize>(n * sizeof(float)));
  if (!f)
    throw std::runtime_error("write failed for " + path);
}

std::vector<uint32_t> read_spv(const std::string &path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f)
    throw std::runtime_error("cannot open SPIR-V " + path);
  auto n = f.tellg();
  if (n <= 0 || (static_cast<size_t>(n) % 4))
    throw std::runtime_error("bad SPIR-V size");
  std::vector<uint32_t> out(static_cast<size_t>(n) / 4);
  f.seekg(0);
  f.read(reinterpret_cast<char *>(out.data()), n);
  if (!f)
    throw std::runtime_error("SPIR-V read failed");
  return out;
}

uint32_t pick_memory_type(const jr::vk::Context &ctx, uint32_t bits,
                          VkMemoryPropertyFlags required,
                          VkMemoryPropertyFlags preferred = 0) {
  VkPhysicalDeviceMemoryProperties mp{};
  vkGetPhysicalDeviceMemoryProperties(ctx.physical_device(), &mp);
  int fallback = -1;
  for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
    if (!(bits & (1u << i)))
      continue;
    auto f = mp.memoryTypes[i].propertyFlags;
    if ((f & required) != required)
      continue;
    if (fallback < 0)
      fallback = static_cast<int>(i);
    if ((f & preferred) == preferred)
      return i;
  }
  if (fallback >= 0)
    return static_cast<uint32_t>(fallback);
  throw std::runtime_error("no compatible Vulkan memory type");
}

Buffer make_buffer(const jr::vk::Context &ctx, VkDeviceSize size,
                   VkBufferUsageFlags usage, VkMemoryPropertyFlags required,
                   bool map = false) {
  Buffer b;
  b.device = ctx.device();
  b.size = size;

  VkBufferCreateInfo ci{};
  ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  ci.size = size;
  ci.usage = usage;
  ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  vk_check(vkCreateBuffer(ctx.device(), &ci, nullptr, &b.buffer),
           "vkCreateBuffer");

  VkMemoryRequirements req{};
  vkGetBufferMemoryRequirements(ctx.device(), b.buffer, &req);

  VkMemoryAllocateInfo ai{};
  ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  ai.allocationSize = req.size;
  ai.memoryTypeIndex = pick_memory_type(ctx, req.memoryTypeBits, required);

  vk_check(vkAllocateMemory(ctx.device(), &ai, nullptr, &b.memory),
           "vkAllocateMemory");
  vk_check(vkBindBufferMemory(ctx.device(), b.buffer, b.memory, 0),
           "vkBindBufferMemory");

  if (map) {
    vk_check(vkMapMemory(ctx.device(), b.memory, 0, size, 0, &b.mapped),
             "vkMapMemory");
  }
  return b;
}

Buffer make_imported_host(const jr::vk::Context &ctx,
                          const std::vector<uint8_t> &data) {
  auto get_props = reinterpret_cast<PFN_vkGetMemoryHostPointerPropertiesEXT>(
      vkGetDeviceProcAddr(ctx.device(), "vkGetMemoryHostPointerPropertiesEXT"));
  if (!get_props)
    throw std::runtime_error("vkGetMemoryHostPointerPropertiesEXT missing");

  const VkDeviceSize align = std::max<VkDeviceSize>(
      ctx.caps().min_imported_host_pointer_alignment, 4096);
  const VkDeviceSize size =
      (static_cast<VkDeviceSize>(data.size()) + align - 1) & ~(align - 1);

  void *host = nullptr;
  if (posix_memalign(&host, static_cast<size_t>(align),
                     static_cast<size_t>(size)) != 0 ||
      !host)
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
  ci.usage =
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  vk_check(vkCreateBuffer(ctx.device(), &ci, nullptr, &b.buffer),
           "vkCreateBuffer(import)");

  VkMemoryRequirements req{};
  vkGetBufferMemoryRequirements(ctx.device(), b.buffer, &req);

  VkMemoryHostPointerPropertiesEXT hp{};
  hp.sType = VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT;
  vk_check(get_props(ctx.device(), kHostHandle, host, &hp),
           "vkGetMemoryHostPointerPropertiesEXT");

  uint32_t bits = hp.memoryTypeBits & req.memoryTypeBits;
  if (!bits)
    throw std::runtime_error("no imported-host memory type");

  VkImportMemoryHostPointerInfoEXT imp{};
  imp.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT;
  imp.handleType = kHostHandle;
  imp.pHostPointer = host;

  VkMemoryAllocateInfo ai{};
  ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  ai.pNext = &imp;
  ai.allocationSize = req.size;
  ai.memoryTypeIndex =
      pick_memory_type(ctx, bits,
                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

  vk_check(vkAllocateMemory(ctx.device(), &ai, nullptr, &b.memory),
           "vkAllocateMemory(import)");
  vk_check(vkBindBufferMemory(ctx.device(), b.buffer, b.memory, 0),
           "vkBindBufferMemory(import)");
  return b;
}

void copy_buffer(const jr::vk::Context &ctx, VkBuffer src, VkBuffer dst,
                 VkDeviceSize size) {
  VkCommandPoolCreateInfo pci{};
  pci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  pci.queueFamilyIndex = ctx.queue_family();

  VkCommandPool pool = VK_NULL_HANDLE;
  vk_check(vkCreateCommandPool(ctx.device(), &pci, nullptr, &pool),
           "vkCreateCommandPool(copy)");

  VkCommandBufferAllocateInfo ai{};
  ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  ai.commandPool = pool;
  ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  ai.commandBufferCount = 1;

  VkCommandBuffer cmd = VK_NULL_HANDLE;
  vk_check(vkAllocateCommandBuffers(ctx.device(), &ai, &cmd),
           "vkAllocateCommandBuffers(copy)");

  VkCommandBufferBeginInfo bi{};
  bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vk_check(vkBeginCommandBuffer(cmd, &bi), "vkBeginCommandBuffer(copy)");

  VkBufferCopy r{};
  r.size = size;
  vkCmdCopyBuffer(cmd, src, dst, 1, &r);
  VkMemoryBarrier ready{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  ready.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  ready.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &ready, 0,
                       nullptr, 0, nullptr);
  vk_check(vkEndCommandBuffer(cmd), "vkEndCommandBuffer(copy)");

  VkFenceCreateInfo fi{};
  fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  VkFence fence = VK_NULL_HANDLE;
  vk_check(vkCreateFence(ctx.device(), &fi, nullptr, &fence),
           "vkCreateFence(copy)");

  VkSubmitInfo si{};
  si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  si.commandBufferCount = 1;
  si.pCommandBuffers = &cmd;
  vk_check(vkQueueSubmit(ctx.queue(), 1, &si, fence), "vkQueueSubmit(copy)");
  vk_check(vkWaitForFences(ctx.device(), 1, &fence, VK_TRUE, UINT64_MAX),
           "vkWaitForFences(copy)");

  vkDestroyFence(ctx.device(), fence, nullptr);
  vkDestroyCommandPool(ctx.device(), pool, nullptr);
}

Buffer
upload_device(const jr::vk::Context &ctx, const void *data, VkDeviceSize size,
              VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) {
  Buffer staging = make_buffer(ctx, size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                   VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                               true);
  std::memcpy(staging.mapped, data, static_cast<size_t>(size));

  Buffer dst = make_buffer(ctx, size, usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                           VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, false);

  copy_buffer(ctx, staging.buffer, dst.buffer, size);
  return dst;
}

Pipeline make_pipeline(const jr::vk::Context &ctx, const char *spv,
                       const std::vector<VkDescriptorBufferInfo> &infos,
                       uint32_t push_size, uint32_t required_subgroup = 32,
                       const uint32_t *specialization = nullptr) {
  Pipeline p;
  p.device = ctx.device();

  static std::map<std::string, std::weak_ptr<PipelineCore>> compiled;
  const auto key = std::to_string(reinterpret_cast<uintptr_t>(ctx.device())) +
                   "|" + spv + "|" + std::to_string(infos.size()) + "|" +
                   std::to_string(push_size) + "|" +
                   std::to_string(required_subgroup) + "|" +
                   (specialization ? std::to_string(*specialization) : "none");
  p.core = compiled[key].lock();
  if (!p.core) {
    p.core = std::make_shared<PipelineCore>();
    p.core->device = ctx.device();
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
    vk_check(
        vkCreateDescriptorSetLayout(ctx.device(), &dl, nullptr, &p.dlayout),
        "vkCreateDescriptorSetLayout");
    p.core->dlayout = p.dlayout;

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
    p.core->layout = p.layout;

    auto code = read_spv(spv);
    VkShaderModuleCreateInfo sm{};
    sm.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    sm.codeSize = code.size() * sizeof(uint32_t);
    sm.pCode = code.data();
    vk_check(vkCreateShaderModule(ctx.device(), &sm, nullptr, &p.shader),
             "vkCreateShaderModule");
    p.core->shader = p.shader;

    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo sg{};
    sg.sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO;
    sg.requiredSubgroupSize = required_subgroup;

    VkPipelineShaderStageCreateInfo stage{};
    stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage.pNext = (ctx.caps().subgroup_size_control &&
                   ctx.caps().min_subgroup_size <= required_subgroup &&
                   ctx.caps().max_subgroup_size >= required_subgroup)
                      ? &sg
                      : nullptr;
    VkSpecializationMapEntry entry{0, 0, 4};
    VkSpecializationInfo spec{1, &entry, 4, specialization};
    if (specialization)
      stage.pSpecializationInfo = &spec;
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = p.shader;
    stage.pName = "main";

    VkComputePipelineCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    ci.stage = stage;
    ci.layout = p.layout;
    vk_check(vkCreateComputePipelines(ctx.device(), VK_NULL_HANDLE, 1, &ci,
                                      nullptr, &p.pipeline),
             "vkCreateComputePipelines");
    p.core->pipeline = p.pipeline;

    compiled[key] = p.core;
  } else {
    p.dlayout = p.core->dlayout;
    p.layout = p.core->layout;
    p.shader = p.core->shader;
    p.pipeline = p.core->pipeline;
  }
  VkDescriptorPoolSize ps{};
  ps.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  ps.descriptorCount = static_cast<uint32_t>(infos.size());

  VkDescriptorPoolCreateInfo dpi{};
  dpi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  dpi.maxSets = 1;
  dpi.poolSizeCount = 1;
  dpi.pPoolSizes = &ps;
  vk_check(vkCreateDescriptorPool(ctx.device(), &dpi, nullptr, &p.pool),
           "vkCreateDescriptorPool");

  VkDescriptorSetAllocateInfo dsai{};
  dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  dsai.descriptorPool = p.pool;
  dsai.descriptorSetCount = 1;
  dsai.pSetLayouts = &p.dlayout;
  vk_check(vkAllocateDescriptorSets(ctx.device(), &dsai, &p.set),
           "vkAllocateDescriptorSets");

  std::vector<VkWriteDescriptorSet> wr(infos.size());
  for (uint32_t i = 0; i < infos.size(); ++i) {
    wr[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    wr[i].dstSet = p.set;
    wr[i].dstBinding = i;
    wr[i].descriptorCount = 1;
    wr[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    wr[i].pBufferInfo = &infos[i];
  }
  vkUpdateDescriptorSets(ctx.device(), static_cast<uint32_t>(wr.size()),
                         wr.data(), 0, nullptr);
  return p;
}

void dispatch(const jr::vk::Context &ctx, Pipeline &p, uint32_t gx,
              const void *push, uint32_t push_size) {
  VkCommandPoolCreateInfo pci{};
  pci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  pci.queueFamilyIndex = ctx.queue_family();

  VkCommandPool pool = VK_NULL_HANDLE;
  vk_check(vkCreateCommandPool(ctx.device(), &pci, nullptr, &pool),
           "vkCreateCommandPool(dispatch)");

  VkCommandBufferAllocateInfo ai{};
  ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  ai.commandPool = pool;
  ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  ai.commandBufferCount = 1;

  VkCommandBuffer cmd = VK_NULL_HANDLE;
  vk_check(vkAllocateCommandBuffers(ctx.device(), &ai, &cmd),
           "vkAllocateCommandBuffers(dispatch)");

  VkCommandBufferBeginInfo bi{};
  bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vk_check(vkBeginCommandBuffer(cmd, &bi), "vkBeginCommandBuffer(dispatch)");

  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p.pipeline);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p.layout, 0, 1,
                          &p.set, 0, nullptr);
  if (push_size) {
    vkCmdPushConstants(cmd, p.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, push_size,
                       push);
  }
  vkCmdDispatch(cmd, gx, 1, 1);
  vk_check(vkEndCommandBuffer(cmd), "vkEndCommandBuffer(dispatch)");

  VkFenceCreateInfo fi{};
  fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  VkFence fence = VK_NULL_HANDLE;
  vk_check(vkCreateFence(ctx.device(), &fi, nullptr, &fence),
           "vkCreateFence(dispatch)");

  VkSubmitInfo si{};
  si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  si.commandBufferCount = 1;
  si.pCommandBuffers = &cmd;
  vk_check(vkQueueSubmit(ctx.queue(), 1, &si, fence),
           "vkQueueSubmit(dispatch)");
  vk_check(vkWaitForFences(ctx.device(), 1, &fence, VK_TRUE, UINT64_MAX),
           "vkWaitForFences(dispatch)");

  vkDestroyFence(ctx.device(), fence, nullptr);
  vkDestroyCommandPool(ctx.device(), pool, nullptr);
}

VkDescriptorBufferInfo info(const Buffer &b,
                            VkDeviceSize range = VK_WHOLE_SIZE) {
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

Tables upload_tables(const jr::vk::Context &ctx) {
  Tables t;

  t.iq2xxs = upload_device(ctx, iq2xxs_grid, sizeof(iq2xxs_grid));
  t.iq2xs = upload_device(ctx, iq2xs_grid, sizeof(iq2xs_grid));
  t.iq2s = upload_device(ctx, iq2s_grid, sizeof(iq2s_grid));
  t.iq3xxs = upload_device(ctx, iq3xxs_grid, sizeof(iq3xxs_grid));
  t.iq3s = upload_device(ctx, iq3s_grid, sizeof(iq3s_grid));

  std::vector<uint32_t> signs32(128);
  for (size_t i = 0; i < signs32.size(); ++i)
    signs32[i] = ksigns_iq2xs[i];
  t.signs =
      upload_device(ctx, signs32.data(), signs32.size() * sizeof(uint32_t));

  std::vector<int32_t> iq4v(16);
  for (size_t i = 0; i < iq4v.size(); ++i)
    iq4v[i] = kvalues_iq4nl[i];
  t.iq4 = upload_device(ctx, iq4v.data(), iq4v.size() * sizeof(int32_t));
  return t;
}

size_t row_bytes_for(int type_id, uint32_t cols) {
  switch (type_id) {
  case 16:
    return static_cast<size_t>(cols / 256u) * 66u;
  case 17:
    return static_cast<size_t>(cols / 256u) * 74u;
  case 18:
    return static_cast<size_t>(cols / 256u) * 98u;
  case 20:
    return static_cast<size_t>(cols / 32u) * 18u;
  case 21:
    return static_cast<size_t>(cols / 256u) * 110u;
  case 22:
    return static_cast<size_t>(cols / 256u) * 82u;
  case 42:
    return static_cast<size_t>(cols / 64u) * 18u;
  default:
    throw std::runtime_error("unsupported expert type " +
                             std::to_string(type_id));
  }
}

size_t q8_bytes(uint32_t n) {
  if (n % 32u)
    throw std::runtime_error("q8_1 width must be divisible by 32");
  return static_cast<size_t>(n / 32u) * 36u;
}

template <typename T> T read_scalar(std::ifstream &f, const char *what) {
  T v{};
  f.read(reinterpret_cast<char *>(&v), sizeof(T));
  if (!f)
    throw std::runtime_error(std::string("bundle read failed: ") + what);
  return v;
}

struct CachePlan {
  uint32_t n_layers = 0;
  uint32_t n_expert = 0;
  uint32_t n_resident = 0;
  uint64_t budget_bytes = 0;
  uint64_t used_bytes = 0;
  std::vector<int32_t> slot; // -1 = imported RAM; >=0 = resident VRAM slot
};

CachePlan read_cache_plan(const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  if (!f)
    throw std::runtime_error("cannot open residency plan " + path);

  char magic[8]{};
  f.read(magic, 8);
  const char expected[8] = {'J', 'R', 'V', 'K', 'C', '9', '\0', '\0'};
  if (!f || std::memcmp(magic, expected, 8) != 0)
    throw std::runtime_error("bad V9 residency plan magic");

  const uint32_t version = read_scalar<uint32_t>(f, "cache version");
  if (version != 1)
    throw std::runtime_error("unsupported V9 residency version");

  CachePlan p;
  p.n_layers = read_scalar<uint32_t>(f, "cache n_layers");
  p.n_expert = read_scalar<uint32_t>(f, "cache n_expert");
  p.n_resident = read_scalar<uint32_t>(f, "cache n_resident");
  (void)read_scalar<uint32_t>(f, "cache flags");
  p.budget_bytes = read_scalar<uint64_t>(f, "cache budget");
  p.used_bytes = read_scalar<uint64_t>(f, "cache used");

  if (p.n_layers == 0 || p.n_expert == 0)
    throw std::runtime_error("invalid V9 residency dimensions");

  const uint64_t cells = uint64_t(p.n_layers) * uint64_t(p.n_expert);
  if (cells > (1ull << 30))
    throw std::runtime_error("unreasonable V9 residency table");

  p.slot.resize(static_cast<size_t>(cells));
  f.read(reinterpret_cast<char *>(p.slot.data()),
         static_cast<std::streamsize>(p.slot.size() * sizeof(int32_t)));
  if (!f)
    throw std::runtime_error("truncated V9 residency table");
  return p;
}

size_t align_up(size_t v, size_t a) { return (v + a - 1) & ~(a - 1); }

#ifdef JR_VK_V8_T16_SHADER
const char *shader_for_quant_type(int32_t type_id) {
  switch (type_id) {
  case 16:
    return JR_VK_V8_T16_SHADER;
  case 17:
    return JR_VK_V8_T17_SHADER;
  case 18:
    return JR_VK_V8_T18_SHADER;
  case 20:
    return JR_VK_V8_T20_SHADER;
  case 21:
    return JR_VK_V8_T21_SHADER;
  case 22:
    return JR_VK_V8_T22_SHADER;
  case 42:
    return JR_VK_V8_T42_SHADER;
  default:
    throw std::runtime_error("V8 has no specialized shader for quant type " +
                             std::to_string(type_id));
  }
}

#endif
} // namespace jr::v10
