#pragma once
#include <vulkan/vulkan.h>
#include <cstdint>
#include <string>
#include <vector>

namespace jr::vk {

struct DeviceSelector {
    uint32_t vendor_id = 0x8086;
    uint32_t device_id = 0xe211; // Intel Arc Pro B60
};

struct DeviceCaps {
    std::string device_name;
    uint32_t vendor_id = 0;
    uint32_t device_id = 0;
    uint32_t api_version = 0;
    uint32_t subgroup_size = 0;
    uint32_t min_subgroup_size = 0;
    uint32_t max_subgroup_size = 0;

    bool buffer_device_address = false;
    bool timeline_semaphore = false;
    bool shader_int64 = false;
    bool shader_float16 = false;
    bool shader_int8 = false;

    bool subgroup_size_control = false;
    bool compute_full_subgroups = false;

    // Vulkan 1.3 integer-dot capability used by V6.
    bool shader_integer_dot_product = false;
    bool intdot_4x8_packed_signed_accelerated = false;
    bool intdot_4x8_packed_mixed_accelerated = false;

    bool ext_external_memory_host = false;
    bool ext_memory_budget = false;
    bool ext_pci_bus_info = false;
    bool ext_subgroup_size_control = false;
    bool khr_cooperative_matrix = false;

    VkDeviceSize min_imported_host_pointer_alignment = 0;
    bool has_pci_bus_info = false;
    uint32_t pci_domain = 0;
    uint32_t pci_bus = 0;
    uint32_t pci_device = 0;
    uint32_t pci_function = 0;
};

class Context {
public:
    explicit Context(DeviceSelector selector = {});
    ~Context();
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;

    VkInstance instance() const { return instance_; }
    VkPhysicalDevice physical_device() const { return physical_device_; }
    VkDevice device() const { return device_; }
    VkQueue queue() const { return queue_; }
    uint32_t queue_family() const { return queue_family_; }
    const DeviceCaps& caps() const { return caps_; }

    uint32_t find_memory_type(uint32_t type_bits,
                              VkMemoryPropertyFlags required,
                              VkMemoryPropertyFlags preferred = 0) const;

private:
    void create_instance();
    void select_physical_device(DeviceSelector selector);
    void query_caps();
    void create_device();

    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    uint32_t queue_family_ = UINT32_MAX;
    DeviceCaps caps_{};
    std::vector<std::string> enabled_extensions_;
};

DeviceSelector selector_from_env();
void vk_check(VkResult result, const char* what);

} // namespace jr::vk
