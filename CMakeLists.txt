cmake_minimum_required(VERSION 3.18)
project(MiniInfer LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 89)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CUDA_FLAGS_RELEASE "-O3")

# 强制 UTF-8 读取源文件，消除 C4819
if(MSVC)
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/utf-8>")
    add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>")
endif()

add_executable(mini_infer
    ${CMAKE_SOURCE_DIR}/main.cu
    ${CMAKE_SOURCE_DIR}/src/tensor.cu
    ${CMAKE_SOURCE_DIR}/src/matmul.cu
    ${CMAKE_SOURCE_DIR}/src/softmax.cu
    ${CMAKE_SOURCE_DIR}/src/relu.cu       
    ${CMAKE_SOURCE_DIR}/src/layers.cu     
)
target_include_directories(mini_infer PRIVATE ${CMAKE_SOURCE_DIR}/include)

find_package(CUDAToolkit REQUIRED)
target_link_libraries(mini_infer PRIVATE CUDA::cudart)