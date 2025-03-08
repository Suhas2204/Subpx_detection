cmake_minimum_required(VERSION 3.10)
project(subpx_test)

# Find dependencies
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(fmt REQUIRED)  # Install via vcpkg/sudo apt-get install libfmt-dev

# Add executable
add_executable(subpx_test
    main.cpp
    libSubpx.cpp
)

# Link libraries
target_link_libraries(subpx_test
    ${OpenCV_LIBS}
    Eigen3::Eigen
    fmt::fmt
)
