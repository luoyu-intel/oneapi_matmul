# CMakeList.txt : CMake project for oneapi_matmul, include source and define
# project specific logic here.
#

# Add source to this project's executable.
file(GLOB SRCS "*.cpp")
file(GLOB HEDS "*.h" "*.hpp")
add_executable (oneapi_matmul ${SRCS} ${HEDS})

target_link_libraries(oneapi_matmul ${ONEAPI_LIBS})
if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET oneapi_matmul PROPERTY CXX_STANDARD 20)
endif()

# TODO: Add tests and install targets if needed.
