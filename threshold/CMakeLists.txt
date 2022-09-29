# cmake needs this line
cmake_minimum_required(VERSION 3.9 FATAL_ERROR)

add_definitions(-std=c++11)

# Define project name
project(adaptiveThreshold)

# Find OpenCV, you may need to set OpenCV_DIR variable
# to the absolute path to the directory containing OpenCVConfig.cmake file
# via the command line or GUI
find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)

if(CMAKE_SIZEOF_VOID_P EQUAL 4)
    message(FATAL_ERROR "32 bit architecture is not supported")
endif()

set(CID ${CMAKE_CXX_COMPILER_ID})
if(WIN32 AND NOT "${CID}" STREQUAL "GNU")
    if("${CID}" STREQUAL "Intel")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:native /fp:fast=2")
    elseif("${CID}" STREQUAL "MSVC")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX2 /fp:fast")
    else()
        message(FATAL_ERROR "Provide your own compile flags.")
    endif()

    # make sure the build is static in Windows (just openmp is not static)
    if("${CID}" STREQUAL "Intel" OR "${CID}" STREQUAL "MSVC")
        foreach(flag_var CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_RELEASE
                CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
            if(${flag_var} MATCHES "/MD")
                string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
            endif()
        endforeach()
    endif()
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math -march=native")
    if("${CID}" STREQUAL "GNU")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static-libstdc++")
    endif()
endif()

# If the package has been found, several variables will
# be set, you can find the full list with descriptions
# in the OpenCVConfig.cmake file.
# Print some message showing some of them
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")


# Declare the executable target built from your sources
add_executable(enhanceScannedImage_exe main.cpp)

# Link your application with OpenCV libraries
target_link_libraries(enhanceScannedImage_exe  OpenMP::OpenMP_CXX ${OpenCV_LIBS})
target_compile_options(enhanceScannedImage_exe PUBLIC ${OpenMP_CXX_FLAGS})
