cmake_minimum_required(VERSION 3.14)
project("ggml-helper")

add_library(whisper STATIC IMPORTED)

set_property(
  TARGET whisper APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE
)

set_target_properties(whisper PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${whisper_ROOT}/lib/static/libwhisper.a"
)
  
set_target_properties(whisper PROPERTIES
  MAP_IMPORTED_CONFIG_MINSIZEREL Release
  MAP_IMPORTED_CONFIG_RELWITHDEBINFO Release
)

target_include_directories(whisper INTERFACE "${whisper_ROOT}/include")

add_library(${PROJECT_NAME} STATIC
    "include/common-ggml.h"
    "src/common-ggml.cpp")

target_compile_features(${PROJECT_NAME}
    PRIVATE
        cxx_std_11
)

# SDL2::SDL2-statics
target_link_libraries(${PROJECT_NAME} PRIVATE whisper)

target_include_directories(${PROJECT_NAME}
    PUBLIC
    "include")