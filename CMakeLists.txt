cmake_minimum_required(VERSION 3.10)
project(MiniCamFPV)

set(CMAKE_CXX_STANDARD 17)

# On cherche juste OpenCV
find_package(OpenCV REQUIRED)

add_executable(minicam main.cpp)
target_link_libraries(minicam ${OpenCV_LIBS})