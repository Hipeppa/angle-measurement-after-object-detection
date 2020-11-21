cmake_minimum_required(VERSION 3.16)
project(bottle2)

set(CMAKE_CXX_STANDARD 14)
# 路径为自己下载的OpenCV路径
set(OpenCV_DIR D://opencv//build//x64//vc14//lib)
set(OpenCV_DIR D://opencv//build//x64//vc14//lib) # 路径为自己下载的OpenCV路径

find_package(OpenCV 3 REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(bottle2 main.cpp)
target_link_libraries(bottle2 ${OpenCV_LIBS})

