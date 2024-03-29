cmake_minimum_required(VERSION 3.18.0)
project("tensor")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

enable_testing()
find_package(GTest REQUIRED)
#find_package(OpenMP REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS}, include)

set(CMAKE_CXX_STANDARD 20)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()



set(CMAKE_CXX_FLAGS "-march=native -Wall -Wextra -std=c++17") 
set(CMAKE_CXX_FLAGS_DEBUG "-g -DDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_BUILD_TYPE=ASAN)

#set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")

include_directories("/")
add_executable(tensor_test tests/tensor_test.cpp)
add_executable(module_test tests/module_test.cpp)
add_executable(forward_test tests/forward_test.cpp)
add_executable(backward_test tests/backward_test.cpp)
add_executable(sin examples/sin.cpp)
add_executable(sinRnn examples/sinRnn.cpp)
add_executable(mnist examples/mnist.cpp)
target_link_libraries(tensor_test ${GTEST_BOTH_LIBRARIES} -lquadmath)
target_link_libraries(forward_test ${GTEST_BOTH_LIBRARIES} -lquadmath)
target_link_libraries(module_test ${GTEST_BOTH_LIBRARIES} -lquadmath)
target_link_libraries(backward_test ${GTEST_BOTH_LIBRARIES} -lquadmath)
target_link_libraries(sin)
target_link_libraries(mnist)
add_test(AllTestsInTensor test_tensor)