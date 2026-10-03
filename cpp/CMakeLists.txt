cmake_minimum_required(VERSION 3.15)
project(bearing_cnn_scorer CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O2")

option(ENABLE_ASAN "Build with AddressSanitizer + UBSan" OFF)
if(ENABLE_ASAN)
  add_compile_options(-fsanitize=address,undefined -g -O1 -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address,undefined)
endif()

find_package(Eigen3 REQUIRED)

add_executable(scorer_main scorer_main.cpp)
target_link_libraries(scorer_main PRIVATE Eigen3::Eigen)

enable_testing()
add_executable(check_scorer tests/test_scorer.cpp)
target_link_libraries(check_scorer PRIVATE Eigen3::Eigen)
add_test(NAME check_scorer COMMAND check_scorer)
