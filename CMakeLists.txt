cmake_minimum_required(VERSION 3.18)

set (CMAKE_CXX_STANDARD 11)

project("whisper.cpp-playground")

add_subdirectory("common")
add_subdirectory("common")


# set (THREADS_PREFER_PTHREAD_FLAG ON)
# find_package(Threads)

# add_library(ggml STATIC IMPORTED)

# set_property(
#   TARGET ggml APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE
# )

# set_target_properties(ggml PROPERTIES
#   IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
#   IMPORTED_LOCATION_RELEASE "${whisper_DIR}/../../../lib/libggml.a"
# )
  
# set_target_properties(ggml PROPERTIES
#   MAP_IMPORTED_CONFIG_MINSIZEREL Release
#   MAP_IMPORTED_CONFIG_RELWITHDEBINFO Release
# )

# target_include_directories(ggml INTERFACE "${whisper_DIR}/../../../include")


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



# find_package(whisper REQUIRED) # PATHS "P:\\Downloads\\cpp-libraries\\portaudio\\build\\cmake\\portaudio")

# if(whisper_FOUND)
#     message(STATUS "whisper_FOUND")
# else()
#     message(STATUS "NO")
# endif()

find_package(SDL2 2.30.6 EXACT REQUIRED) # PATHS "P:\\Downloads\\cpp-libraries\\portaudio\\build\\cmake\\portaudio")

if(SDL2_FOUND)
    message(STATUS "SDL2_FOUND")
else()
    message(STATUS "NO")
endif()

add_executable(${PROJECT_NAME}
    "src/main.cpp"
)

target_compile_features(${PROJECT_NAME}
    PRIVATE
        cxx_std_11
)




target_link_libraries(${PROJECT_NAME} PRIVATE SDL2::SDL2-static whisper)
target_link_options(${PROJECT_NAME} PRIVATE "-static")

