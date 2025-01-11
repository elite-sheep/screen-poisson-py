cmake_minimum_required(VERSION 3.20)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

project(ScreenPoissonPy LANGUAGES CXX CUDA)

find_package(Python 3.10 COMPONENTS Interpreter Development REQUIRED)

find_package(CUDAToolkit 11.5 REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES 35)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -rdc=true")
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ext/nanobind)
set(NANOBIND_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/ext/nanobind)
set(SCREENPOISSON_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${NANOBIND_ROOT}/include
    ${SCREENPOISSON_SOURCE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS})

message(STATUS "CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")

add_library(screenpoissoncuda STATIC src/Backend.cpp
                                     src/Backend.hpp
                                     src/BackendCUDA.cu
                                     src/BackendCUDA.hpp
                                     src/Defs.cpp
                                     src/Defs.hpp
                                     src/Solver.cpp
                                     src/Solver.hpp)
# set_property(TARGET screenpoissoncuda PROPERTY CUDA_ARCHITECTURES native)
set_target_properties(screenpoissoncuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                                                   CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(screenpoissoncuda PROPERTIES COMPILE_OPTIONS -Xcompiler)
target_compile_options(screenpoissoncuda PRIVATE /MP)
target_compile_features(screenpoissoncuda PUBLIC cxx_std_11)
target_link_libraries(screenpoissoncuda PRIVATE CUDA::cudart 
                                                CUDA::cudart_static
                                                CUDA::cuda_driver)

nanobind_add_module(screenpoissonpy src/screen_poisson_py.cpp)
set_property(TARGET screenpoissonpy PROPERTY CUDA_ARCHITECTURES native)
target_link_libraries(screenpoissonpy PRIVATE screenpoissoncuda)