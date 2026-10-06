#include "jrstrata/vk/vk_context.hpp"

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>

namespace jr::vk {
namespace {

uint32_t parse_hex_u32(const std::string& s) {
    size_t pos = 0;
    unsigned long v = std::stoul(s, &pos, 16);
    if (pos != s.size() || v > 0xffffffffUL) throw std::runtime_error("invalid hex value: " + s);
    return static_cast<uint32_t>(v);
}

std::string hex4(uint32_t v) {
    std::ostringstream oss;
    oss << std::hex << std::setfill('0') << std::setw(4) << v;
    return oss.str();
}

std::set<std::string> enumerate_extensions(VkPhysicalDevice pd) {
    uint32_t count = 0;
    vk_check(vkEnumerateDeviceExtensionProperties(pd, nullptr, &count, nullptr),
             "vkEnumerateDeviceExtensionProperties(count)");
    std::vector<VkExtensionProperties> props(count);
    vk_check(vkEnumerateDeviceExtensionProperties(pd, nullptr, &count, props.data()),
             "vkEnumerateDeviceExtensionProperties(list)");
    std::set<std::string> out;
    for (const auto& p : props) out.emplace(p.extensionName);
    return out;
}

bool has_ext(const std::set<std::string>& exts, const char* name) {
    return exts.find(name) != exts.end();
}

} // namespace

void vk_check(VkResult result, const char* what) {
    if (result != VK_SUCCESS) {
        std::ostringstream oss;
        oss << what << " failed with VkResult " << static_cast<int>(result);
        throw std::runtime_error(oss.str());
    }
}

DeviceSelector selector_from_env() {
    DeviceSelector s{};
    const char* env = std::getenv("JR_VK_DEVICE");
    if (!env || !*env) return s;
    std::string v(env);
    auto colon = v.find(':');
    if (colon == std::string::npos)
        throw std::runtime_error("JR_VK_DEVICE must be VENDOR:DEVICE in hex, e.g. 8086:e211");
    s.vendor_id = parse_hex_u32(v.substr(0, colon));
    s.device_id = parse_hex_u32(v.substr(colon + 1));
    return s;
}

Context::Context(DeviceSelector selector) {
    try {
        create_instance();
        select_physical_device(selector);
        query_caps();
        create_device();
    } catch (...) {
        if (device_) vkDestroyDevice(device_, nullptr);
        if (instance_) vkDestroyInstance(instance_, nullptr);
        throw;
    }
}

Context::~Context() {
    if (device_) {
        vkDeviceWaitIdle(device_);
        vkDestroyDevice(device_, nullptr);
    }
    if (instance_) vkDestroyInstance(instance_, nullptr);
}

void Context::create_instance() {
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "JR-Strata-Vulkan";
    app.applicationVersion = VK_MAKE_VERSION(0, 6, 0);
    app.pEngineName = "JR-Strata-Vulkan";
    app.engineVersion = VK_MAKE_VERSION(0, 6, 0);
    app.apiVersion = VK_API_VERSION_1_3;

    VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ci.pApplicationInfo = &app;
    vk_check(vkCreateInstance(&ci, nullptr, &instance_), "vkCreateInstance");
}

void Context::select_physical_device(DeviceSelector selector) {
    uint32_t count = 0;
    vk_check(vkEnumeratePhysicalDevices(instance_, &count, nullptr), "vkEnumeratePhysicalDevices(count)");
    if (count == 0) throw std::runtime_error("no Vulkan physical devices found");

    std::vector<VkPhysicalDevice> devs(count);
    vk_check(vkEnumeratePhysicalDevices(instance_, &count, devs.data()), "vkEnumeratePhysicalDevices(list)");

    std::cout << "JR-VK: enumerating " << count << " Vulkan device(s)\n";
    for (uint32_t i = 0; i < count; ++i) {
        VkPhysicalDeviceProperties p{};
        vkGetPhysicalDeviceProperties(devs[i], &p);
        std::cout << "  [" << i << "] " << p.deviceName
                  << " vendor=" << hex4(p.vendorID)
                  << " device=" << hex4(p.deviceID)
                  << " api=" << VK_VERSION_MAJOR(p.apiVersion) << "."
                  << VK_VERSION_MINOR(p.apiVersion) << "."
                  << VK_VERSION_PATCH(p.apiVersion) << "\n";
        if (p.vendorID == selector.vendor_id && p.deviceID == selector.device_id)
            physical_device_ = devs[i];
    }

    if (!physical_device_)
        throw std::runtime_error("requested Vulkan device " + hex4(selector.vendor_id) + ":" +
                                 hex4(selector.device_id) + " was not found");

    uint32_t qcount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &qcount, nullptr);
    std::vector<VkQueueFamilyProperties> qprops(qcount);
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &qcount, qprops.data());

    for (uint32_t i = 0; i < qcount; ++i) {
        if ((qprops[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && !(qprops[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
            queue_family_ = i;
            break;
        }
    }
    if (queue_family_ == UINT32_MAX) {
        for (uint32_t i = 0; i < qcount; ++i) {
            if (qprops[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                queue_family_ = i;
                break;
            }
        }
    }
    if (queue_family_ == UINT32_MAX) throw std::runtime_error("selected device has no compute queue");
}

void Context::query_caps() {
    const auto exts = enumerate_extensions(physical_device_);
    caps_.ext_external_memory_host = has_ext(exts, VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME);
    caps_.ext_memory_budget = has_ext(exts, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
    caps_.ext_pci_bus_info = has_ext(exts, VK_EXT_PCI_BUS_INFO_EXTENSION_NAME);
    caps_.ext_subgroup_size_control = has_ext(exts, VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME);
    caps_.khr_cooperative_matrix = has_ext(exts, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);

    VkPhysicalDeviceSubgroupProperties subgroup{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
    VkPhysicalDeviceSubgroupSizeControlProperties subgroup_ctl_props{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES};
    VkPhysicalDeviceExternalMemoryHostPropertiesEXT host_props{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT};
    VkPhysicalDevicePCIBusInfoPropertiesEXT pci{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PCI_BUS_INFO_PROPERTIES_EXT};
    VkPhysicalDeviceShaderIntegerDotProductProperties intdot_props{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_PROPERTIES};
    VkPhysicalDeviceProperties2 props2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};

    props2.pNext = &subgroup;
    void** tail = &subgroup.pNext;

    if (caps_.ext_subgroup_size_control) {
        *tail = &subgroup_ctl_props;
        tail = &subgroup_ctl_props.pNext;
    }
    if (caps_.ext_external_memory_host) {
        *tail = &host_props;
        tail = &host_props.pNext;
    }
    if (caps_.ext_pci_bus_info) {
        *tail = &pci;
        tail = &pci.pNext;
    }

    // Core Vulkan 1.3 property structure. B60 exposes API >= 1.3.
    *tail = &intdot_props;
    intdot_props.pNext = nullptr;

    vkGetPhysicalDeviceProperties2(physical_device_, &props2);

    caps_.device_name = props2.properties.deviceName;
    caps_.vendor_id = props2.properties.vendorID;
    caps_.device_id = props2.properties.deviceID;
    caps_.api_version = props2.properties.apiVersion;
    caps_.subgroup_size = subgroup.subgroupSize;

    if (caps_.ext_subgroup_size_control) {
        caps_.min_subgroup_size = subgroup_ctl_props.minSubgroupSize;
        caps_.max_subgroup_size = subgroup_ctl_props.maxSubgroupSize;
    }
    if (caps_.ext_external_memory_host)
        caps_.min_imported_host_pointer_alignment = host_props.minImportedHostPointerAlignment;

    if (caps_.ext_pci_bus_info) {
        caps_.has_pci_bus_info = true;
        caps_.pci_domain = pci.pciDomain;
        caps_.pci_bus = pci.pciBus;
        caps_.pci_device = pci.pciDevice;
        caps_.pci_function = pci.pciFunction;
    }

    caps_.intdot_4x8_packed_signed_accelerated =
        intdot_props.integerDotProduct4x8BitPackedSignedAccelerated == VK_TRUE;
    caps_.intdot_4x8_packed_mixed_accelerated =
        intdot_props.integerDotProduct4x8BitPackedMixedSignednessAccelerated == VK_TRUE;

    VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};

    f2.pNext = &f12;
    f12.pNext = &f13;
    f13.pNext = nullptr;
    vkGetPhysicalDeviceFeatures2(physical_device_, &f2);

    caps_.shader_int64 = f2.features.shaderInt64 == VK_TRUE;
    caps_.buffer_device_address = f12.bufferDeviceAddress == VK_TRUE;
    caps_.timeline_semaphore = f12.timelineSemaphore == VK_TRUE;
    caps_.shader_float16 = f12.shaderFloat16 == VK_TRUE;
    caps_.shader_int8 = f12.shaderInt8 == VK_TRUE;

    caps_.subgroup_size_control = f13.subgroupSizeControl == VK_TRUE;
    caps_.compute_full_subgroups = f13.computeFullSubgroups == VK_TRUE;
    caps_.shader_integer_dot_product = f13.shaderIntegerDotProduct == VK_TRUE;
}

void Context::create_device() {
    if (!caps_.buffer_device_address) throw std::runtime_error("bufferDeviceAddress is required");
    if (!caps_.timeline_semaphore) throw std::runtime_error("timelineSemaphore is required");
    if (!caps_.ext_external_memory_host) throw std::runtime_error("VK_EXT_external_memory_host is required");

    enabled_extensions_.push_back(VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME);
    if (caps_.ext_memory_budget) enabled_extensions_.push_back(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
    if (caps_.ext_pci_bus_info) enabled_extensions_.push_back(VK_EXT_PCI_BUS_INFO_EXTENSION_NAME);
    if (caps_.ext_subgroup_size_control) enabled_extensions_.push_back(VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME);

    std::vector<const char*> ext_names;
    for (auto& s : enabled_extensions_) ext_names.push_back(s.c_str());

    float priority = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = queue_family_;
    qci.queueCount = 1;
    qci.pQueuePriorities = &priority;

    VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    f13.subgroupSizeControl = caps_.subgroup_size_control ? VK_TRUE : VK_FALSE;
    f13.computeFullSubgroups = caps_.compute_full_subgroups ? VK_TRUE : VK_FALSE;
    f13.shaderIntegerDotProduct = caps_.shader_integer_dot_product ? VK_TRUE : VK_FALSE;

    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    f12.bufferDeviceAddress = VK_TRUE;
    f12.timelineSemaphore = VK_TRUE;
    f12.pNext = &f13;

    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.pNext = &f12;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = static_cast<uint32_t>(ext_names.size());
    dci.ppEnabledExtensionNames = ext_names.data();

    vk_check(vkCreateDevice(physical_device_, &dci, nullptr, &device_), "vkCreateDevice");
    vkGetDeviceQueue(device_, queue_family_, 0, &queue_);
    if (!queue_) throw std::runtime_error("vkGetDeviceQueue returned null");
}

uint32_t Context::find_memory_type(uint32_t type_bits,
                                   VkMemoryPropertyFlags required,
                                   VkMemoryPropertyFlags preferred) const {
    VkPhysicalDeviceMemoryProperties mp{};
    vkGetPhysicalDeviceMemoryProperties(physical_device_, &mp);
    int fallback = -1;
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
        if (!(type_bits & (1u << i))) continue;
        const auto flags = mp.memoryTypes[i].propertyFlags;
        if ((flags & required) != required) continue;
        if (fallback < 0) fallback = static_cast<int>(i);
        if ((flags & preferred) == preferred) return i;
    }
    if (fallback >= 0) return static_cast<uint32_t>(fallback);
    throw std::runtime_error("no compatible Vulkan memory type found");
}

} // namespace jr::vk
