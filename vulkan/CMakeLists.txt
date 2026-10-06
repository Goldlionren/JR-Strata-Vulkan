cmake_minimum_required(VERSION 3.24)

project(jr_strata_vulkan_v0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Vulkan REQUIRED)
find_program(GLSLANG_VALIDATOR glslangValidator REQUIRED)

function(jr_compile_shader target_name shader_name)
    set(src "${CMAKE_CURRENT_SOURCE_DIR}/shaders/${shader_name}")
    set(out "${CMAKE_CURRENT_BINARY_DIR}/shaders/${shader_name}.spv")
    add_custom_command(
        OUTPUT "${out}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/shaders"
        COMMAND "${GLSLANG_VALIDATOR}"
                -V
                --target-env vulkan1.3
                "${src}"
                -o "${out}"
        DEPENDS "${src}"
        COMMENT "Compiling ${shader_name} -> SPIR-V"
        VERBATIM
    )
    add_custom_target(${target_name} DEPENDS "${out}")
    set(${target_name}_SPV "${out}" PARENT_SCOPE)
endfunction()

jr_compile_shader(jr_vk_vector_shader vector_add.comp)
jr_compile_shader(jr_vk_host_copy_shader host_copy.comp)
jr_compile_shader(jr_vk_read_scalar_shader host_read_scalar.comp)
jr_compile_shader(jr_vk_read_vec4_shader host_read_vec4.comp)
jr_compile_shader(jr_vk_read_u2_shader host_read_uvec4x2.comp)
jr_compile_shader(jr_vk_read_u4_shader host_read_uvec4x4.comp)
jr_compile_shader(jr_vk_iq3xxs_shader iq3_xxs_dequant.comp)

# V2 full-expert path.
jr_compile_shader(jr_vk_expert_dequant_shader expert_dequant.comp)
jr_compile_shader(jr_vk_expert_gemv_shader expert_gemv.comp)
jr_compile_shader(jr_vk_expert_swiglu_shader expert_swiglu.comp)

add_library(jr_vk_runtime STATIC
    src/vk_context.cpp
)

target_include_directories(jr_vk_runtime PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_link_libraries(jr_vk_runtime PUBLIC Vulkan::Vulkan)
target_compile_options(jr_vk_runtime PRIVATE -Wall -Wextra -Wpedantic)

add_executable(jr-vk-v0 src/main.cpp)
add_dependencies(jr-vk-v0 jr_vk_vector_shader)
target_link_libraries(jr-vk-v0 PRIVATE jr_vk_runtime)
target_compile_definitions(jr-vk-v0 PRIVATE
    JR_VK_VECTOR_SHADER="${jr_vk_vector_shader_SPV}"
)

add_executable(jr-vk-memprobe src/memprobe.cpp)
add_dependencies(jr-vk-memprobe jr_vk_host_copy_shader)
target_link_libraries(jr-vk-memprobe PRIVATE jr_vk_runtime)
target_compile_definitions(jr-vk-memprobe PRIVATE
    JR_VK_HOST_COPY_SHADER="${jr_vk_host_copy_shader_SPV}"
)
target_compile_options(jr-vk-memprobe PRIVATE -Wall -Wextra -Wpedantic)

add_executable(jr-vk-memprobe-v04 src/memprobe_v04.cpp)
add_dependencies(jr-vk-memprobe-v04
    jr_vk_read_scalar_shader
    jr_vk_read_vec4_shader
    jr_vk_read_u2_shader
    jr_vk_read_u4_shader
)
target_link_libraries(jr-vk-memprobe-v04 PRIVATE jr_vk_runtime)
target_compile_definitions(jr-vk-memprobe-v04 PRIVATE
    JR_VK_READ_SCALAR_SHADER="${jr_vk_read_scalar_shader_SPV}"
    JR_VK_READ_VEC4_SHADER="${jr_vk_read_vec4_shader_SPV}"
    JR_VK_READ_U2_SHADER="${jr_vk_read_u2_shader_SPV}"
    JR_VK_READ_U4_SHADER="${jr_vk_read_u4_shader_SPV}"
)
target_compile_options(jr-vk-memprobe-v04 PRIVATE -Wall -Wextra -Wpedantic)

add_executable(jr-vk-iq3xxs-parity src/iq3xxs_parity.cpp)
add_dependencies(jr-vk-iq3xxs-parity jr_vk_iq3xxs_shader)
target_link_libraries(jr-vk-iq3xxs-parity PRIVATE jr_vk_runtime)
target_include_directories(jr-vk-iq3xxs-parity PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../include"
    "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/ggml"
)
target_compile_definitions(jr-vk-iq3xxs-parity PRIVATE
    JR_VK_IQ3XXS_SHADER="${jr_vk_iq3xxs_shader_SPV}"
)
target_compile_options(jr-vk-iq3xxs-parity PRIVATE -Wall -Wextra -Wpedantic)

add_executable(jr-vk-full-expert
    src/full_expert.cpp
)
add_dependencies(jr-vk-full-expert
    jr_vk_expert_dequant_shader
    jr_vk_expert_gemv_shader
    jr_vk_expert_swiglu_shader
)
target_link_libraries(jr-vk-full-expert PRIVATE jr_vk_runtime)
target_include_directories(jr-vk-full-expert PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/ggml"
)
target_compile_definitions(jr-vk-full-expert PRIVATE
    JR_VK_EXPERT_DEQUANT_SHADER="${jr_vk_expert_dequant_shader_SPV}"
    JR_VK_EXPERT_GEMV_SHADER="${jr_vk_expert_gemv_shader_SPV}"
    JR_VK_EXPERT_SWIGLU_SHADER="${jr_vk_expert_swiglu_shader_SPV}"
)
target_compile_options(jr-vk-full-expert PRIVATE -Wall -Wextra -Wpedantic)
