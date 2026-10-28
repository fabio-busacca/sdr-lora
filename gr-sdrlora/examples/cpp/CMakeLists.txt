# Builds hello_lora against an installed sdrlora-phy library:
#   cmake -S . -B build && cmake --build build && ./build/hello_lora
# Pass -DCMAKE_PREFIX_PATH=<prefix> if the library is not under /usr/local.
cmake_minimum_required(VERSION 3.16)
project(hello_lora CXX)

find_package(sdrlora-phy 2.0 REQUIRED)

add_executable(hello_lora hello_lora.cc)
target_link_libraries(hello_lora PRIVATE sdrlora::sdrlora-phy)
