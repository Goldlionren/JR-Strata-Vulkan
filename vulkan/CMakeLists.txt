cmake_minimum_required(VERSION 3.24)
project(jr_strata_vulkan_v0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Vulkan REQUIRED)
find_program(GLSLANG_VALIDATOR glslangValidator REQUIRED)

set(SHADER_SRC "${CMAKE_CURRENT_SOURCE_DIR}/shaders/vector_add.comp")
set(SHADER_OUT "${CMAKE_CURRENT_BINARY_DIR}/shaders/vector_add.comp.spv")

add_custom_command(
    OUTPUT "${SHADER_OUT}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/shaders"
    COMMAND "${GLSLANG_VALIDATOR}" -V --target-env vulkan1.3 "${SHADER_SRC}" -o "${SHADER_OUT}"
    DEPENDS "${SHADER_SRC}"
    COMMENT "Compiling vector_add.comp -> SPIR-V"
    VERBATIM)

add_custom_target(jr_vk_shaders ALL DEPENDS "${SHADER_OUT}")

add_executable(jr-vk-v0
    src/main.cpp
    src/vk_context.cpp)
add_dependencies(jr-vk-v0 jr_vk_shaders)

target_include_directories(jr-vk-v0 PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(jr-vk-v0 PRIVATE Vulkan::Vulkan)
target_compile_definitions(jr-vk-v0 PRIVATE JR_VK_VECTOR_SHADER="${SHADER_OUT}")
target_compile_options(jr-vk-v0 PRIVATE -Wall -Wextra -Wpedantic)
