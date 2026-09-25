cmake_minimum_required(VERSION 3.15)
project(am05_osarsa_seq_core LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(Threads REQUIRED)
find_path(EIGEN3_INCLUDE_DIR eigen3/Eigen/Dense
  HINTS /usr/local/include /opt/homebrew/include /usr/include)
if(NOT EIGEN3_INCLUDE_DIR)
  message(FATAL_ERROR "Eigen3 headers not found. Install Eigen3 or pass -DEIGEN3_INCLUDE_DIR=<include-dir>.")
endif()
add_executable(osarsa-seq
  src/main.cpp
  src/core/common.cpp
  src/core/utils.cpp
  src/parser/parser_light.cpp
  src/MDP/Backward_induction.cpp
  src/POMDP/belief.cpp
  src/decPOMDP/core/support.cpp
  src/decPOMDP/core/decision_rule.cpp
  src/decPOMDP/core/occupancy_state_base.cpp
  src/decPOMDP/core/distribution.cpp
  src/decPOMDP/sequential/occupancy_state.cpp
  src/decPOMDP/sequential/oSarsa.cpp
  src/decPOMDP/sequential/value_representation/Q_representation.cpp
  src/decPOMDP/sequential/value_representation/mdp_solution.cpp
  src/decPOMDP/sequential/value_representation/reward_shaping.cpp)
target_include_directories(osarsa-seq PRIVATE "${EIGEN3_INCLUDE_DIR}")
target_link_libraries(osarsa-seq PRIVATE Threads::Threads)
