message(STATUS "[INFO] Building gsa_prefetch for device: ${RUNTIME_ENVIRONMENT}")

# 查找必要的包
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# 查找PyTorch路径
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import torch; import os; print(os.path.dirname(os.path.abspath(torch.__file__)))"
    OUTPUT_VARIABLE PYTORCH_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE PYTORCH_RESULT
)

if(NOT PYTORCH_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to find PyTorch installation")
endif()

# 设置基础编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fopenmp -march=native")
set(CXX11_ABI "1")

# 根据设备类型配置
set(INCLUDE_DIRS
    ${PYTORCH_PATH}/include/torch/csrc/api/include
    ${PYTORCH_PATH}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/ucm/store
)

set(LIBRARY_DIRS
    ${PYTORCH_PATH}/lib
    /usr/local/lib
)

set(LIBRARIES
    torch
    c10
    torch_cpu
    torch_python
    gomp
    pthread
    storeintf
)

# NPU特殊配置
if(RUNTIME_ENVIRONMENT STREQUAL "ascend")
    message(STATUS "Configuring for NPU/Ascend device")

    # 查找torch_npu路径
    execute_process(
        COMMAND ${Python_EXECUTABLE} -c "import torch_npu; import os; print(os.path.dirname(os.path.abspath(torch_npu.__file__)))"
        OUTPUT_VARIABLE PYTORCH_NPU_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE NPU_RESULT
    )

    if(NPU_RESULT EQUAL 0)
        message(STATUS "Found torch_npu at: ${PYTORCH_NPU_PATH}")
        list(INSERT INCLUDE_DIRS 0 ${PYTORCH_NPU_PATH}/include)
        list(INSERT LIBRARY_DIRS 0 ${PYTORCH_NPU_PATH}/lib)
        list(INSERT LIBRARIES 0 torch_npu)
        set(CXX11_ABI "0")
    else()
        message(WARNING "torch_npu not found, but RUNTIME_ENVIRONMENT is set to ascend")
    endif()
else()
    set(CUDA_ROOT "/usr/local/cuda/" CACHE PATH "Path to CUDA root directory")
    set(CMAKE_CUDA_COMPILER ${CUDA_ROOT}/bin/nvcc)
    set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90)
    enable_language(CUDA)
    list(INSERT INCLUDE_DIRS 0 ${CUDA_ROOT}/include)
    list(INSERT LIBRARY_DIRS 0 ${CUDA_ROOT}/lib64)
    add_library(cudakernel OBJECT ${CMAKE_CURRENT_SOURCE_DIR}/include/cuda_sm_kernel.cu)
    target_compile_options(cudakernel PRIVATE
        --diag-suppress=128 --diag-suppress=2417 --diag-suppress=2597
        -Wall -fPIC
    )
    list(INSERT LIBRARIES 0 cudakernel)
endif()

# 设置CXX11_ABI宏
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=${CXX11_ABI}")

# 查找OpenMP
find_package(OpenMP REQUIRED)

# 定义源文件
set(SOURCES
    src/pybinds.cpp
    src/kvcache_pre.cpp
)

# 创建pybind11模块
pybind11_add_module(gsa_prefetch ${SOURCES})

# 设置头文件目录
target_include_directories(gsa_prefetch PRIVATE ${INCLUDE_DIRS})

# 设置库文件目录
target_link_directories(gsa_prefetch PRIVATE ${LIBRARY_DIRS})

# 链接库
target_link_libraries(gsa_prefetch PRIVATE ${LIBRARIES})

# 链接OpenMP
if(OpenMP_CXX_FOUND)
    target_link_libraries(gsa_prefetch PRIVATE OpenMP::OpenMP_CXX)
endif()

file(RELATIVE_PATH INSTALL_REL_PATH
     ${CMAKE_SOURCE_DIR}
     ${CMAKE_CURRENT_SOURCE_DIR}
)
install(TARGETS gsa_prefetch LIBRARY DESTINATION ${INSTALL_REL_PATH} COMPONENT ucm)
