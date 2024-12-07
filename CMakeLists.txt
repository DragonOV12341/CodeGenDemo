# project config
###################################################################
cmake_minimum_required(VERSION 3.15.0)
project(KernelCodeGen LANGUAGES CXX C)    # delete CUDA

set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)
set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")

set(LLVM_BUILD_LIBRARY_DIR "~/rocm-llvm-project/build/lib")
message(STATUS "LLVM_BUILD_LIBRARY_DIR ${LLVM_BUILD_LIBRARY_DIR}")

# set(MLIR_CONFIG_FILE_DIR /home/pangyunfei/rocm-llvm-project/build/lib/cmake/mlir)
# message(STATUS "MLIR_CONFIG_FILE_DIR ${MLIR_CONFIG_FILE_DIR}")

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  find_package(MLIR REQUIRED CONFIG PATHS ${CMAKE_SOURCE_DIR}/cmake)

  message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
  message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

  set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
  set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
  set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

  list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

  include(TableGen)
  include(AddLLVM)
  include(AddMLIR)
  include(HandleLLVMOptions)
else()
  # Build via external projects mechanism
  set(MLIR_MAIN_SRC_DIR ${LLVM_MAIN_SRC_DIR}/../mlir)
  set(MLIR_INCLUDE_DIR ${MLIR_MAIN_SRC_DIR}/include)
  set(MLIR_GENERATED_INCLUDE_DIR ${LLVM_BINARY_DIR}/tools/mlir/include)
  set(MLIR_INCLUDE_DIRS "${MLIR_INCLUDE_DIR};${MLIR_GENERATED_INCLUDE_DIR}")
endif()
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(KernelCodeGen_SOURCE_DIR ${PROJECT_SOURCE_DIR})
set(KernelCodeGen_BINARY_DIR ${PROJECT_BINARY_DIR})
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${KernelCodeGen_SOURCE_DIR}/include)
include_directories(${KernelCodeGen_BINARY_DIR}/include)
include_directories(${KernelCodeGen_SOURCE_DIR}/build/include)

link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})
message("add-def:${LLVM_DEFINITIONS}")
add_compile_options(-g)
add_definitions(-DGEN_PASS_DEF_CONVERTMEMREFTOLLVMPTR)

# optional config
###################################################################
# list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
# include(config)
SET(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

# add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(test)
add_subdirectory(include/Conversion)