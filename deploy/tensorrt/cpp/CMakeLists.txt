cmake_minimum_required(VERSION 3.16)
project(light_rcnn_postprocess LANGUAGES CXX)

add_library(light_rcnn_nms STATIC deterministic_nms.cpp)
target_compile_features(light_rcnn_nms PUBLIC cxx_std_17)
target_include_directories(light_rcnn_nms PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
