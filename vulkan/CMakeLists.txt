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
    set(shader_dependencies "${src}")
    if(shader_name MATCHES "^v(10|11)_")
        if(shader_name MATCHES "^v11_expert_t")
            list(APPEND shader_dependencies "${CMAKE_CURRENT_SOURCE_DIR}/shaders/v11_expert_batch.glsl")
        endif()
        if(shader_name MATCHES "^v11_")
            list(APPEND shader_dependencies "${CMAKE_CURRENT_SOURCE_DIR}/shaders/v11_common.glsl")
        else()
            list(APPEND shader_dependencies "${CMAKE_CURRENT_SOURCE_DIR}/shaders/v10_common.glsl")
        endif()
    endif()
    add_custom_command(
        OUTPUT "${out}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/shaders"
        COMMAND "${GLSLANG_VALIDATOR}"
                -V
                --target-env vulkan1.3
                "${src}"
                -o "${out}"
        DEPENDS ${shader_dependencies}
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
jr_compile_shader(jr_vk_fused_gemv_shader fused_expert_gemv.comp)
jr_compile_shader(jr_vk_q8_quant_shader q8_1_quant.comp)
jr_compile_shader(jr_vk_swiglu_q8_shader swiglu_q8_1.comp)
jr_compile_shader(jr_vk_q8_gemv_shader fused_q8_expert_gemv.comp)
jr_compile_shader(jr_vk_weighted_accum_shader weighted_accum.comp)
jr_compile_shader(jr_vk_v5_grouped_gemv_shader v5_grouped_q8_gemv.comp)
jr_compile_shader(jr_vk_v6_hwdot_shader v6_grouped_q8_gemv_hwdot.comp)
jr_compile_shader(jr_vk_v7_quadrow_shader v7_grouped_q8_gemv_quadrow.comp)
jr_compile_shader(jr_vk_v8_t16_shader v8_t16_q8_gemv.comp)
jr_compile_shader(jr_vk_v8_t17_shader v8_t17_q8_gemv.comp)
jr_compile_shader(jr_vk_v8_t18_shader v8_t18_q8_gemv.comp)
jr_compile_shader(jr_vk_v8_t20_shader v8_t20_q8_gemv.comp)
jr_compile_shader(jr_vk_v8_t21_shader v8_t21_q8_gemv.comp)
jr_compile_shader(jr_vk_v8_t22_shader v8_t22_q8_gemv.comp)
jr_compile_shader(jr_vk_v8_t42_shader v8_t42_q8_gemv.comp)
jr_compile_shader(jr_vk_v5_grouped_swiglu_shader v5_grouped_swiglu_q8.comp)
jr_compile_shader(jr_vk_v5_combine_shader v5_route_combine.comp)

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


# V3: production-shaped expert path. Quantized weights are consumed directly
# from imported host memory; no full F32 weight materialization.
add_executable(jr-vk-fused-expert
    src/fused_expert.cpp
)
add_dependencies(jr-vk-fused-expert
    jr_vk_fused_gemv_shader
    jr_vk_expert_swiglu_shader
)
target_link_libraries(jr-vk-fused-expert PRIVATE jr_vk_runtime)
target_include_directories(jr-vk-fused-expert PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/ggml"
)
target_compile_definitions(jr-vk-fused-expert PRIVATE
    JR_VK_FUSED_GEMV_SHADER="${jr_vk_fused_gemv_shader_SPV}"
    JR_VK_EXPERT_SWIGLU_SHADER="${jr_vk_expert_swiglu_shader_SPV}"
)
target_compile_options(jr-vk-fused-expert PRIVATE -Wall -Wextra -Wpedantic)


# V4: routed expert engine.
# The routing decision is an input contract, matching Strata native_expert_grouped:
# selected expert IDs + normalized route weights are already known.
# V4 executes top-10 experts across mixed VRAM/imported-RAM tiers, with q8_1
# activations and weighted accumulation.
add_executable(jr-vk-routed-expert
    src/routed_expert.cpp
)
add_dependencies(jr-vk-routed-expert
    jr_vk_q8_quant_shader
    jr_vk_swiglu_q8_shader
    jr_vk_q8_gemv_shader
    jr_vk_weighted_accum_shader
)
target_link_libraries(jr-vk-routed-expert PRIVATE jr_vk_runtime)
target_include_directories(jr-vk-routed-expert PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/ggml"
)
target_compile_definitions(jr-vk-routed-expert PRIVATE
    JR_VK_Q8_QUANT_SHADER="${jr_vk_q8_quant_shader_SPV}"
    JR_VK_SWIGLU_Q8_SHADER="${jr_vk_swiglu_q8_shader_SPV}"
    JR_VK_Q8_GEMV_SHADER="${jr_vk_q8_gemv_shader_SPV}"
    JR_VK_WEIGHTED_ACCUM_SHADER="${jr_vk_weighted_accum_shader_SPV}"
)
target_compile_options(jr-vk-routed-expert PRIVATE -Wall -Wextra -Wpedantic)


# V5: production-shaped routed engine.
#
# One persistent command buffer executes the entire top-k group:
#   x -> q8_1
#   grouped gate+up (all selected experts in one dispatch)
#   grouped SwiGLU -> q8_1
#   grouped down (all selected experts in one dispatch)
#   route-weighted combine
#
# Selected expert blobs are packed once into two route arenas:
#   tier 0 = imported system RAM
#   tier 1 = device-local VRAM
# and addressed by a compact expert metadata table.
add_executable(jr-vk-routed-engine-v5
    src/routed_engine_v5.cpp
)
add_dependencies(jr-vk-routed-engine-v5
    jr_vk_q8_quant_shader
    jr_vk_v5_grouped_gemv_shader
    jr_vk_v5_grouped_swiglu_shader
    jr_vk_v5_combine_shader
)
target_link_libraries(jr-vk-routed-engine-v5 PRIVATE jr_vk_runtime)
target_include_directories(jr-vk-routed-engine-v5 PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/ggml"
)
target_compile_definitions(jr-vk-routed-engine-v5 PRIVATE
    JR_VK_Q8_QUANT_SHADER="${jr_vk_q8_quant_shader_SPV}"
    JR_VK_V5_GROUPED_GEMV_SHADER="${jr_vk_v5_grouped_gemv_shader_SPV}"
    JR_VK_V5_GROUPED_SWIGLU_SHADER="${jr_vk_v5_grouped_swiglu_shader_SPV}"
    JR_VK_V5_COMBINE_SHADER="${jr_vk_v5_combine_shader_SPV}"
)
target_compile_options(jr-vk-routed-engine-v5 PRIVATE -Wall -Wextra -Wpedantic)


# V6: hardware packed integer-dot specialization.
#
# Same persistent/grouped engine as V5, but the hot quantized dot path can use
# GL_EXT_integer_dot_product -> OpSDotKHR when the selected Vulkan device
# advertises shaderIntegerDotProduct. Auto mode requires the driver to also
# report integerDotProduct4x8BitPackedSignedAccelerated.
add_executable(jr-vk-routed-engine-v6
    src/routed_engine_v6.cpp
)
add_dependencies(jr-vk-routed-engine-v6
    jr_vk_q8_quant_shader
    jr_vk_v5_grouped_gemv_shader
    jr_vk_v6_hwdot_shader
    jr_vk_v5_grouped_swiglu_shader
    jr_vk_v5_combine_shader
)
target_link_libraries(jr-vk-routed-engine-v6 PRIVATE jr_vk_runtime)
target_include_directories(jr-vk-routed-engine-v6 PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/ggml"
)
target_compile_definitions(jr-vk-routed-engine-v6 PRIVATE
    JR_VK_Q8_QUANT_SHADER="${jr_vk_q8_quant_shader_SPV}"
    JR_VK_V5_SOFT_GEMV_SHADER="${jr_vk_v5_grouped_gemv_shader_SPV}"
    JR_VK_V6_HWDOT_SHADER="${jr_vk_v6_hwdot_shader_SPV}"
    JR_VK_V5_GROUPED_SWIGLU_SHADER="${jr_vk_v5_grouped_swiglu_shader_SPV}"
    JR_VK_V5_COMBINE_SHADER="${jr_vk_v5_combine_shader_SPV}"
)
target_compile_options(jr-vk-routed-engine-v6 PRIVATE -Wall -Wextra -Wpedantic)


# V7: B60/Xe2 production specialization.
#
# Four output rows share one 32-lane workgroup; each row is computed by 8 lanes.
# This increases per-lane work, reuses quant block scale/metadata, cuts row
# workgroup count by 4x, and retains accelerated packed signed 4x8 OpSDot.
add_executable(jr-vk-routed-engine-v7
    src/routed_engine_v7.cpp
)
add_dependencies(jr-vk-routed-engine-v7
    jr_vk_q8_quant_shader
    jr_vk_v7_quadrow_shader
    jr_vk_v5_grouped_swiglu_shader
    jr_vk_v5_combine_shader
)
target_link_libraries(jr-vk-routed-engine-v7 PRIVATE jr_vk_runtime)
target_include_directories(jr-vk-routed-engine-v7 PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/ggml"
)
target_compile_definitions(jr-vk-routed-engine-v7 PRIVATE
    JR_VK_Q8_QUANT_SHADER="${jr_vk_q8_quant_shader_SPV}"
    JR_VK_V7_QUADROW_SHADER="${jr_vk_v7_quadrow_shader_SPV}"
    JR_VK_V5_GROUPED_SWIGLU_SHADER="${jr_vk_v5_grouped_swiglu_shader_SPV}"
    JR_VK_V5_COMBINE_SHADER="${jr_vk_v5_combine_shader_SPV}"
)
target_compile_options(jr-vk-routed-engine-v7 PRIVATE -Wall -Wextra -Wpedantic)


# V8: format-specialized Xe2 expert kernels.
#
# All selected experts in a production layer share one gate/up quant format and
# one down quant format. V7 still carried all seven decoders in a single shader.
# V8 compiles one shader per real production quant type and picks the two
# required pipelines once per route bundle.
add_executable(jr-vk-routed-engine-v8
    src/routed_engine_v8.cpp
)
add_dependencies(jr-vk-routed-engine-v8
    jr_vk_q8_quant_shader
    jr_vk_v8_t16_shader
    jr_vk_v8_t17_shader
    jr_vk_v8_t18_shader
    jr_vk_v8_t20_shader
    jr_vk_v8_t21_shader
    jr_vk_v8_t22_shader
    jr_vk_v8_t42_shader
    jr_vk_v5_grouped_swiglu_shader
    jr_vk_v5_combine_shader
)
target_link_libraries(jr-vk-routed-engine-v8 PRIVATE jr_vk_runtime)
target_include_directories(jr-vk-routed-engine-v8 PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/ggml"
)
target_compile_definitions(jr-vk-routed-engine-v8 PRIVATE
    JR_VK_Q8_QUANT_SHADER="${jr_vk_q8_quant_shader_SPV}"
    JR_VK_V8_T16_SHADER="${jr_vk_v8_t16_shader_SPV}"
    JR_VK_V8_T17_SHADER="${jr_vk_v8_t17_shader_SPV}"
    JR_VK_V8_T18_SHADER="${jr_vk_v8_t18_shader_SPV}"
    JR_VK_V8_T20_SHADER="${jr_vk_v8_t20_shader_SPV}"
    JR_VK_V8_T21_SHADER="${jr_vk_v8_t21_shader_SPV}"
    JR_VK_V8_T22_SHADER="${jr_vk_v8_t22_shader_SPV}"
    JR_VK_V8_T42_SHADER="${jr_vk_v8_t42_shader_SPV}"
    JR_VK_V5_GROUPED_SWIGLU_SHADER="${jr_vk_v5_grouped_swiglu_shader_SPV}"
    JR_VK_V5_COMBINE_SHADER="${jr_vk_v5_combine_shader_SPV}"
)
target_compile_options(jr-vk-routed-engine-v8 PRIVATE -Wall -Wextra -Wpedantic)


# V9: profile-driven production cache policy / routed execution.
#
# V9 keeps V8's format-specialized Xe2 kernels and adds a whole-model
# residency plan derived from Strata expert-profile.bin + native_experts.txt.
# The route bundle no longer decides VRAM/RAM placement: the engine resolves
# every routed (layer, expert) through the V9 residency table.
add_executable(jr-vk-routed-engine-v9
    src/routed_engine_v9.cpp
)
add_dependencies(jr-vk-routed-engine-v9
    jr_vk_q8_quant_shader
    jr_vk_v8_t16_shader
    jr_vk_v8_t17_shader
    jr_vk_v8_t18_shader
    jr_vk_v8_t20_shader
    jr_vk_v8_t21_shader
    jr_vk_v8_t22_shader
    jr_vk_v8_t42_shader
    jr_vk_v5_grouped_swiglu_shader
    jr_vk_v5_combine_shader
)
target_link_libraries(jr-vk-routed-engine-v9 PRIVATE jr_vk_runtime)
target_include_directories(jr-vk-routed-engine-v9 PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/ggml"
)
target_compile_definitions(jr-vk-routed-engine-v9 PRIVATE
    JR_VK_Q8_QUANT_SHADER="${jr_vk_q8_quant_shader_SPV}"
    JR_VK_V8_T16_SHADER="${jr_vk_v8_t16_shader_SPV}"
    JR_VK_V8_T17_SHADER="${jr_vk_v8_t17_shader_SPV}"
    JR_VK_V8_T18_SHADER="${jr_vk_v8_t18_shader_SPV}"
    JR_VK_V8_T20_SHADER="${jr_vk_v8_t20_shader_SPV}"
    JR_VK_V8_T21_SHADER="${jr_vk_v8_t21_shader_SPV}"
    JR_VK_V8_T22_SHADER="${jr_vk_v8_t22_shader_SPV}"
    JR_VK_V8_T42_SHADER="${jr_vk_v8_t42_shader_SPV}"
    JR_VK_V5_GROUPED_SWIGLU_SHADER="${jr_vk_v5_grouped_swiglu_shader_SPV}"
    JR_VK_V5_COMBINE_SHADER="${jr_vk_v5_combine_shader_SPV}"
)
target_compile_options(jr-vk-routed-engine-v9 PRIVATE -Wall -Wextra -Wpedantic)

# V10: complete Swift token engine. V8/V9 routed expert shaders stay unchanged.
foreach(shader v10_dense.comp v10_ops.comp v10_gdn.comp v10_attn.comp v10_qround.comp v10_dense_small.comp v10_router.comp v10_hc.comp)
    string(REPLACE ".comp" "" name "${shader}")
    jr_compile_shader(jr_vk_${name}_shader ${shader})
endforeach()

add_executable(jr-vk-runtime-v10 src/runtime_v10.cpp)
target_link_libraries(jr-vk-runtime-v10 PRIVATE jr_vk_runtime)
target_include_directories(jr-vk-runtime-v10 PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../include"
    "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/ggml")
add_dependencies(jr-vk-runtime-v10
    jr_vk_v10_dense_shader jr_vk_v10_ops_shader jr_vk_v10_gdn_shader jr_vk_v10_attn_shader jr_vk_v10_qround_shader jr_vk_v10_dense_small_shader jr_vk_v10_router_shader jr_vk_v10_hc_shader
    jr_vk_q8_quant_shader jr_vk_v5_grouped_swiglu_shader jr_vk_v5_combine_shader
    jr_vk_v8_t16_shader jr_vk_v8_t17_shader jr_vk_v8_t18_shader
    jr_vk_v8_t20_shader jr_vk_v8_t21_shader jr_vk_v8_t22_shader jr_vk_v8_t42_shader)
target_compile_definitions(jr-vk-runtime-v10 PRIVATE
    JR_VK_V10_DENSE_SHADER="${jr_vk_v10_dense_shader_SPV}"
    JR_VK_V10_QROUND_SHADER="${jr_vk_v10_qround_shader_SPV}"
    JR_VK_V10_SMALL_SHADER="${jr_vk_v10_dense_small_shader_SPV}"
    JR_VK_V10_ROUTER_SHADER="${jr_vk_v10_router_shader_SPV}"
    JR_VK_V10_HC_SHADER="${jr_vk_v10_hc_shader_SPV}"
    JR_VK_V10_OPS_SHADER="${jr_vk_v10_ops_shader_SPV}"
    JR_VK_V10_GDN_SHADER="${jr_vk_v10_gdn_shader_SPV}"
    JR_VK_V10_ATTN_SHADER="${jr_vk_v10_attn_shader_SPV}"
    JR_VK_Q8_QUANT_SHADER="${jr_vk_q8_quant_shader_SPV}"
    JR_VK_V5_GROUPED_SWIGLU_SHADER="${jr_vk_v5_grouped_swiglu_shader_SPV}"
    JR_VK_V5_COMBINE_SHADER="${jr_vk_v5_combine_shader_SPV}"
    JR_VK_V8_T16_SHADER="${jr_vk_v8_t16_shader_SPV}"
    JR_VK_V8_T17_SHADER="${jr_vk_v8_t17_shader_SPV}"
    JR_VK_V8_T18_SHADER="${jr_vk_v8_t18_shader_SPV}"
    JR_VK_V8_T20_SHADER="${jr_vk_v8_t20_shader_SPV}"
    JR_VK_V8_T21_SHADER="${jr_vk_v8_t21_shader_SPV}"
    JR_VK_V8_T22_SHADER="${jr_vk_v8_t22_shader_SPV}"
    JR_VK_V8_T42_SHADER="${jr_vk_v8_t42_shader_SPV}"
    JR_V10_FRONTEND="${CMAKE_CURRENT_SOURCE_DIR}/../tools/vulkan_runtime_v10.py")
add_executable(jr-vk-server-v10 src/server_v10.cpp)
add_dependencies(jr-vk-server-v10 jr-vk-runtime-v10)
target_compile_definitions(jr-vk-server-v10 PRIVATE
    JR_V10_FRONTEND="${CMAKE_CURRENT_SOURCE_DIR}/../tools/vulkan_runtime_v10.py")

target_compile_options(jr-vk-runtime-v10 PRIVATE -Wall -Wextra -Wpedantic -Wno-missing-field-initializers)

# V11 has independent shader artifacts, preserving the frozen V10 executable.
foreach(shader v11_dense.comp v11_ops.comp v11_gdn.comp v11_attn.comp v11_qround.comp v11_dense_small.comp v11_router.comp v11_hc.comp v11_dense_batch.comp v11_gdn_batch.comp v11_dense_small_batch.comp)
    string(REPLACE ".comp" "" name "${shader}")
    jr_compile_shader(jr_vk_${name}_shader ${shader})
endforeach()
add_executable(jr-vk-runtime-v11 src/runtime_v11.cpp)
target_link_libraries(jr-vk-runtime-v11 PRIVATE jr_vk_runtime)
target_include_directories(jr-vk-runtime-v11 PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../include"
    "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/ggml")
add_dependencies(jr-vk-runtime-v11
    jr_vk_v11_dense_batch_shader jr_vk_v11_gdn_batch_shader jr_vk_v11_dense_small_batch_shader
    jr_vk_v11_dense_shader jr_vk_v11_ops_shader jr_vk_v11_gdn_shader jr_vk_v11_attn_shader jr_vk_v11_qround_shader jr_vk_v11_dense_small_shader jr_vk_v11_router_shader jr_vk_v11_hc_shader
    jr_vk_q8_quant_shader jr_vk_v5_grouped_swiglu_shader jr_vk_v5_combine_shader
    jr_vk_v8_t16_shader jr_vk_v8_t17_shader jr_vk_v8_t18_shader
    jr_vk_v8_t20_shader jr_vk_v8_t21_shader jr_vk_v8_t22_shader jr_vk_v8_t42_shader)
target_compile_definitions(jr-vk-runtime-v11 PRIVATE
    JR_V11_SMALL_BATCH_SHADER="${jr_vk_v11_dense_small_batch_shader_SPV}"
    JR_V11_DENSE_BATCH_SHADER="${jr_vk_v11_dense_batch_shader_SPV}"
    JR_V11_GDN_BATCH_SHADER="${jr_vk_v11_gdn_batch_shader_SPV}"
    JR_VK_V10_DENSE_SHADER="${jr_vk_v11_dense_shader_SPV}"
    JR_VK_V10_QROUND_SHADER="${jr_vk_v11_qround_shader_SPV}"
    JR_VK_V10_SMALL_SHADER="${jr_vk_v11_dense_small_shader_SPV}"
    JR_VK_V10_ROUTER_SHADER="${jr_vk_v11_router_shader_SPV}"
    JR_VK_V10_HC_SHADER="${jr_vk_v11_hc_shader_SPV}"
    JR_VK_V10_OPS_SHADER="${jr_vk_v11_ops_shader_SPV}"
    JR_VK_V10_GDN_SHADER="${jr_vk_v11_gdn_shader_SPV}"
    JR_VK_V10_ATTN_SHADER="${jr_vk_v11_attn_shader_SPV}"
    JR_VK_Q8_QUANT_SHADER="${jr_vk_q8_quant_shader_SPV}"
    JR_VK_V5_GROUPED_SWIGLU_SHADER="${jr_vk_v5_grouped_swiglu_shader_SPV}"
    JR_VK_V5_COMBINE_SHADER="${jr_vk_v5_combine_shader_SPV}"
    JR_VK_V8_T16_SHADER="${jr_vk_v8_t16_shader_SPV}"
    JR_VK_V8_T17_SHADER="${jr_vk_v8_t17_shader_SPV}"
    JR_VK_V8_T18_SHADER="${jr_vk_v8_t18_shader_SPV}"
    JR_VK_V8_T20_SHADER="${jr_vk_v8_t20_shader_SPV}"
    JR_VK_V8_T21_SHADER="${jr_vk_v8_t21_shader_SPV}"
    JR_VK_V8_T22_SHADER="${jr_vk_v8_t22_shader_SPV}"
    JR_VK_V8_T42_SHADER="${jr_vk_v8_t42_shader_SPV}"
    JR_V11_FRONTEND="${CMAKE_CURRENT_SOURCE_DIR}/../tools/vulkan_runtime_v11.py")
add_executable(jr-vk-server-v11 src/server_v11.cpp)
add_dependencies(jr-vk-server-v11 jr-vk-runtime-v11)
target_compile_definitions(jr-vk-server-v11 PRIVATE
    JR_V11_FRONTEND="${CMAKE_CURRENT_SOURCE_DIR}/../tools/vulkan_runtime_v11.py")

target_compile_options(jr-vk-runtime-v11 PRIVATE -Wall -Wextra -Wpedantic -Wno-missing-field-initializers)

# V11 groups duplicate routes only within a target window; V9 remains unchanged.
jr_compile_shader(jr_vk_v11_expert_t16_shader v11_expert_t16.comp)
add_dependencies(jr-vk-runtime-v11 jr_vk_v11_expert_t16_shader)
target_compile_definitions(jr-vk-runtime-v11 PRIVATE JR_V11_EXPERT_T16_SHADER="${jr_vk_v11_expert_t16_shader_SPV}")
jr_compile_shader(jr_vk_v11_expert_t17_shader v11_expert_t17.comp)
add_dependencies(jr-vk-runtime-v11 jr_vk_v11_expert_t17_shader)
target_compile_definitions(jr-vk-runtime-v11 PRIVATE JR_V11_EXPERT_T17_SHADER="${jr_vk_v11_expert_t17_shader_SPV}")
jr_compile_shader(jr_vk_v11_expert_t18_shader v11_expert_t18.comp)
add_dependencies(jr-vk-runtime-v11 jr_vk_v11_expert_t18_shader)
target_compile_definitions(jr-vk-runtime-v11 PRIVATE JR_V11_EXPERT_T18_SHADER="${jr_vk_v11_expert_t18_shader_SPV}")
jr_compile_shader(jr_vk_v11_expert_t20_shader v11_expert_t20.comp)
add_dependencies(jr-vk-runtime-v11 jr_vk_v11_expert_t20_shader)
target_compile_definitions(jr-vk-runtime-v11 PRIVATE JR_V11_EXPERT_T20_SHADER="${jr_vk_v11_expert_t20_shader_SPV}")
jr_compile_shader(jr_vk_v11_expert_t21_shader v11_expert_t21.comp)
add_dependencies(jr-vk-runtime-v11 jr_vk_v11_expert_t21_shader)
target_compile_definitions(jr-vk-runtime-v11 PRIVATE JR_V11_EXPERT_T21_SHADER="${jr_vk_v11_expert_t21_shader_SPV}")
jr_compile_shader(jr_vk_v11_expert_t22_shader v11_expert_t22.comp)
add_dependencies(jr-vk-runtime-v11 jr_vk_v11_expert_t22_shader)
target_compile_definitions(jr-vk-runtime-v11 PRIVATE JR_V11_EXPERT_T22_SHADER="${jr_vk_v11_expert_t22_shader_SPV}")
jr_compile_shader(jr_vk_v11_expert_t42_shader v11_expert_t42.comp)
add_dependencies(jr-vk-runtime-v11 jr_vk_v11_expert_t42_shader)
target_compile_definitions(jr-vk-runtime-v11 PRIVATE JR_V11_EXPERT_T42_SHADER="${jr_vk_v11_expert_t42_shader_SPV}")
jr_compile_shader(jr_vk_v11_expert_group_shader v11_expert_group.comp)
add_dependencies(jr-vk-runtime-v11 jr_vk_v11_expert_group_shader)
target_compile_definitions(jr-vk-runtime-v11 PRIVATE JR_V11_EXPERT_GROUP_SHADER="${jr_vk_v11_expert_group_shader_SPV}")

jr_compile_shader(jr_vk_v11_hc_batch_shader v11_hc_batch.comp)
add_dependencies(jr-vk-runtime-v11 jr_vk_v11_hc_batch_shader)
target_compile_definitions(jr-vk-runtime-v11 PRIVATE JR_V11_HC_BATCH_SHADER="${jr_vk_v11_hc_batch_shader_SPV}")

add_executable(jr-v11-hc-check ../tools/vulkan_v11_hc_check.cpp)
target_link_libraries(jr-v11-hc-check PRIVATE jr_vk_runtime)
target_include_directories(jr-v11-hc-check PRIVATE src ../include ../third_party/ggml)
target_compile_options(jr-v11-hc-check PRIVATE -Wno-missing-field-initializers)
add_dependencies(jr-v11-hc-check jr_vk_v11_ops_shader)
target_compile_definitions(jr-v11-hc-check PRIVATE JR_HC_OPS="${jr_vk_v11_ops_shader_SPV}")
add_dependencies(jr-v11-hc-check jr_vk_v11_dense_small_shader)
target_compile_definitions(jr-v11-hc-check PRIVATE JR_HC_SMALL="${jr_vk_v11_dense_small_shader_SPV}")
add_dependencies(jr-v11-hc-check jr_vk_v11_hc_shader)
target_compile_definitions(jr-v11-hc-check PRIVATE JR_HC_UP="${jr_vk_v11_hc_shader_SPV}")
add_dependencies(jr-v11-hc-check jr_vk_v11_dense_small_batch_shader)
target_compile_definitions(jr-v11-hc-check PRIVATE JR_HC_BATCH_DOWN="${jr_vk_v11_dense_small_batch_shader_SPV}")
add_dependencies(jr-v11-hc-check jr_vk_v11_hc_batch_shader)
target_compile_definitions(jr-v11-hc-check PRIVATE JR_HC_BATCH_UP="${jr_vk_v11_hc_batch_shader_SPV}")

jr_compile_shader(jr_vk_v11_mtp_attn_shader v11_mtp_attn.comp)
add_dependencies(jr-vk-runtime-v11 jr_vk_v11_mtp_attn_shader)
target_compile_definitions(jr-vk-runtime-v11 PRIVATE JR_V11_MTP_ATTN_SHADER="${jr_vk_v11_mtp_attn_shader_SPV}")


# Batch independent frame normalization and Q8 rounding without changing arithmetic.
jr_compile_shader(jr_vk_v11_norm_batch_shader v11_norm_batch.comp)
add_dependencies(jr-vk-runtime-v11 jr_vk_v11_norm_batch_shader)
target_compile_definitions(jr-vk-runtime-v11 PRIVATE JR_V11_NORM_BATCH_SHADER="${jr_vk_v11_norm_batch_shader_SPV}")
jr_compile_shader(jr_vk_v11_qround_batch_shader v11_qround_batch.comp)
add_dependencies(jr-vk-runtime-v11 jr_vk_v11_qround_batch_shader)
target_compile_definitions(jr-vk-runtime-v11 PRIVATE JR_V11_QROUND_BATCH_SHADER="${jr_vk_v11_qround_batch_shader_SPV}")
add_dependencies(jr-v11-hc-check jr_vk_v11_norm_batch_shader)
target_compile_definitions(jr-v11-hc-check PRIVATE JR_HC_BATCH_NORM="${jr_vk_v11_norm_batch_shader_SPV}")

jr_compile_shader(jr_vk_v11_gdn_prepare_batch_shader v11_gdn_prepare_batch.comp)
add_dependencies(jr-vk-runtime-v11 jr_vk_v11_gdn_prepare_batch_shader)
target_compile_definitions(jr-vk-runtime-v11 PRIVATE JR_V11_GDN_PREPARE_BATCH_SHADER="${jr_vk_v11_gdn_prepare_batch_shader_SPV}")

jr_compile_shader(jr_vk_v11_qsa_ops_batch_shader v11_qsa_ops_batch.comp)
add_dependencies(jr-vk-runtime-v11 jr_vk_v11_qsa_ops_batch_shader)
target_compile_definitions(jr-vk-runtime-v11 PRIVATE JR_V11_QSA_OPS_BATCH_SHADER="${jr_vk_v11_qsa_ops_batch_shader_SPV}")

jr_compile_shader(jr_vk_v11_attn_batch_shader v11_attn_batch.comp)
add_dependencies(jr-vk-runtime-v11 jr_vk_v11_attn_batch_shader)
target_compile_definitions(jr-vk-runtime-v11 PRIVATE JR_V11_ATTN_BATCH_SHADER="${jr_vk_v11_attn_batch_shader_SPV}")

jr_compile_shader(jr_vk_v11_router_batch_shader v11_router_batch.comp)
add_dependencies(jr-vk-runtime-v11 jr_vk_v11_router_batch_shader)
target_compile_definitions(jr-vk-runtime-v11 PRIVATE JR_V11_ROUTER_BATCH_SHADER="${jr_vk_v11_router_batch_shader_SPV}")

jr_compile_shader(jr_vk_v11_swiglu_batch_shader v11_swiglu_batch.comp)
add_dependencies(jr-vk-runtime-v11 jr_vk_v11_swiglu_batch_shader)
target_compile_definitions(jr-vk-runtime-v11 PRIVATE JR_V11_SWIGLU_BATCH_SHADER="${jr_vk_v11_swiglu_batch_shader_SPV}")

jr_compile_shader(jr_vk_v11_combine_batch_shader v11_combine_batch.comp)
add_dependencies(jr-vk-runtime-v11 jr_vk_v11_combine_batch_shader)
target_compile_definitions(jr-vk-runtime-v11 PRIVATE JR_V11_COMBINE_BATCH_SHADER="${jr_vk_v11_combine_batch_shader_SPV}")

jr_compile_shader(jr_vk_v11_attn_tile_shader v11_attn_tile.comp)
add_executable(jr-v11-attention-check ../tools/vulkan_v11_attention_check.cpp)
target_link_libraries(jr-v11-attention-check PRIVATE jr_vk_runtime)
target_include_directories(jr-v11-attention-check PRIVATE src ../include ../third_party/ggml)
target_compile_options(jr-v11-attention-check PRIVATE -Wno-missing-field-initializers)
add_dependencies(jr-v11-attention-check jr_vk_v11_attn_batch_shader jr_vk_v11_attn_tile_shader)
target_compile_definitions(jr-v11-attention-check PRIVATE
    JR_ATTN_REFERENCE="${jr_vk_v11_attn_batch_shader_SPV}"
    JR_ATTN_TILE="${jr_vk_v11_attn_tile_shader_SPV}")

jr_compile_shader(jr_vk_v11_qsa_select_batch_shader v11_qsa_select_batch.comp)
add_dependencies(jr-v11-attention-check jr_vk_v11_qsa_select_batch_shader jr_vk_v11_ops_shader)
target_compile_definitions(jr-v11-attention-check PRIVATE JR_QSA_REFERENCE_OPS="${jr_vk_v11_ops_shader_SPV}" JR_QSA_SELECT="${jr_vk_v11_qsa_select_batch_shader_SPV}")

add_dependencies(jr-vk-runtime-v11 jr_vk_v11_qsa_select_batch_shader)
target_compile_definitions(jr-vk-runtime-v11 PRIVATE JR_V11_QSA_SELECT_BATCH_SHADER="${jr_vk_v11_qsa_select_batch_shader_SPV}")

add_executable(jr-v11-expert-check ../tools/vulkan_v11_expert_check.cpp)
target_link_libraries(jr-v11-expert-check PRIVATE jr_vk_runtime)
target_include_directories(jr-v11-expert-check PRIVATE src ../include ../third_party/ggml)
target_compile_options(jr-v11-expert-check PRIVATE -Wno-missing-field-initializers)
foreach(type 16 17 18 20 21 22 42)
    add_dependencies(jr-v11-expert-check jr_vk_v11_expert_t${type}_shader)
    target_compile_definitions(jr-v11-expert-check PRIVATE JR_TEST_T${type}="${jr_vk_v11_expert_t${type}_shader_SPV}")
endforeach()
