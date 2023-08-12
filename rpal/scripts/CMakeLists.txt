cmake_minimum_required(VERSION 3.12)
project(image_buffer)

find_package(pybind11 REQUIRED)
find_package(OpenCV REQUIRED)
pybind11_add_module(image_buffer MODULE image_buffer.cc)
target_link_libraries(image_buffer PRIVATE ${OpenCV_LIBS})
