cmake_minimum_required(VERSION 4.1)
project(Neuromorphic_PE_Simulation)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(SYSTEMC_HOME "/opt/systemc/" CACHE PATH "Path to SystemC install prefix")

set(SYSTEMC_INCLUDE_DIR ${SYSTEMC_HOME}/include)
set(SYSTEMC_LIBRARY_DIR ${SYSTEMC_HOME}/lib)

add_executable(NoC_Simulation
        src/main.cpp
)

target_include_directories(NoC_Simulation PRIVATE ${SYSTEMC_INCLUDE_DIR})
target_link_libraries(NoC_Simulation ${SYSTEMC_LIBRARY_DIR}/libsystemc.so)
target_include_directories(NoC_Simulation PUBLIC ${SYSTEMC_INCLUDE_DIR})
target_include_directories(NoC_Simulation PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
message(STATUS "SystemC Home: ${SYSTEMC_HOME}")