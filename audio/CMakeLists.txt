cmake_minimum_required(VERSION 3.18)
project("audio")


find_package(SDL2 2.30.6 EXACT REQUIRED) # PATHS "P:\\Downloads\\cpp-libraries\\portaudio\\build\\cmake\\portaudio")

if(SDL2_FOUND)
    message(STATUS "SDL2_FOUND")
else()
    message(STATUS "NO")
endif()

add_library(${PROJECT_NAME} STATIC
    "include/common-sdl.h"
    "src/common-sdl.cpp")

target_compile_features(${PROJECT_NAME}
    PRIVATE
        cxx_std_11
)

target_link_libraries(${PROJECT_NAME} PRIVATE SDL2::SDL2-static)

target_include_directories(${PROJECT_NAME}
    PUBLIC
    "include")