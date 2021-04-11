cmake_minimum_required(VERSION 3.18.0)
project("tensor")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

enable_testing()
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS}, include)

set(CXX_STANDARD 17)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-march=native -Wall -Wextra -Werror -std=c++17") 
set(CMAKE_CXX_FLAGS_DEBUG "-g -DDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_BUILD_TYPE=ASAN)


include_directories("/")
add_executable(tensor_test tests/tensor_test.cpp)
add_executable(forward_test tests/forward_test.cpp)
add_executable(backward_test tests/backward_test.cpp)
add_executable(sin examples/sin.cpp)
target_link_libraries(tensor_test ${GTEST_BOTH_LIBRARIES} -lquadmath)
target_link_libraries(forward_test ${GTEST_BOTH_LIBRARIES} -lquadmath)
target_link_libraries(backward_test ${GTEST_BOTH_LIBRARIES} -lquadmath)
target_link_libraries(sin)
add_test(AllTestsInTensor test_tensor)