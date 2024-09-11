cmake_minimum_required(VERSION 3.12.2)
message(STATUS "CMake version: ${CMAKE_VERSION}")

project(signals)

add_subdirectory(cpp)
add_subdirectory(python)

add_subdirectory(cpp/pybind11)

pybind11_add_module(module_name cpp/example.cpp)