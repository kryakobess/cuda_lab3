cmake_minimum_required(VERSION 3.30)
project(lab_3 CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_executable(lab_3 main.cu)

set_target_properties(lab_3 PROPERTIES
                      CUDA_SEPARABLE_COMPILATION ON)
