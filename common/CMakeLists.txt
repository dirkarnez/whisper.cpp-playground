cmake_minimum_required(VERSION 3.14)
project("common")

add_library(${PROJECT_NAME} STATIC
    "include/common.h"
    "src/common.cpp")

target_compile_features(${PROJECT_NAME}
    PRIVATE
        cxx_std_11
)

# SDL2::SDL2-statics
# target_link_libraries(${PROJECT_NAME} PRIVATE whisper ggml-helper)

    
target_include_directories(${PROJECT_NAME}
    PUBLIC
    "include")