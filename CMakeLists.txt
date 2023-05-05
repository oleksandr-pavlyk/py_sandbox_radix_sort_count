cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

if (${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.24")
  cmake_policy(SET CMP0135 NEW)
endif()

project(radix-sort-count
  LANGUAGES CXX
  DESCRIPTION "Toy extensions to experiment with function from DPL"
  )

find_package(IntelSYCL REQUIRED)

if (NOT DEFINED DPCTL_MODULE_PATH)
  if (DEFINED ENV{DPCTL_MODULE_PATH})
    set(DPCTL_MODULE_PATH $ENV{DPCTL_MODULE_PATH})
  else ()
    message(FATAL_ERROR "Specify DPCTL_MODULE_PATH, either via cmake or as environment varibale")
  endif()
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${DPCTL_MODULE_PATH})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Fetch pybind11
include(FetchContent)
FetchContent_Declare(
  pybind11
  URL https://github.com/pybind/pybind11/archive/refs/tags/v2.10.2.tar.gz
  URL_HASH SHA256=93bd1e625e43e03028a3ea7389bba5d3f9f2596abc074b068e70f4ef9b1314ae
  )
FetchContent_MakeAvailable(pybind11)

find_package(PythonExtensions REQUIRED)
find_package(Dpctl REQUIRED)

set(_borrowed_includes ${Dpctl_INCLUDE_DIRS}/../../dpctl/tensor/libtensor/include)

set(py_module_name _radix)
set(_radix_sources )
list(APPEND _radix_sources
  src/_py.cpp
  )
pybind11_add_module(${py_module_name}
  MODULE
  ${_radix_sources}
  )
target_compile_options(${py_module_name} PUBLIC -O3 -fno-approx-func)
target_include_directories(${py_module_name} PUBLIC src)
target_include_directories(${py_module_name} PUBLIC ${Dpctl_INCLUDE_DIRS})
target_include_directories(${py_module_name} PUBLIC ${_borrowed_includes})
target_link_options(${py_module_name} PRIVATE -fsycl-device-code-split=per_kernel)

add_sycl_to_target(TARGET ${py_module_name} SOURCES ${_radix_sources})

install(
  TARGETS ${py_module_name}
  DESTINATION radix
  )

set(ignoreMe "${SKBUILD}")
