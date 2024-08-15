cmake_minimum_required(VERSION 3.18)
project("app" CXX)

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


find_package(SDL2 2.30.6 EXACT REQUIRED) # PATHS "P:\\Downloads\\cpp-libraries\\portaudio\\build\\cmake\\portaudio")

if(SDL2_FOUND)
    message(STATUS "SDL2_FOUND")
else()
    message(STATUS "NO")
endif()





find_package(ZLIB REQUIRED)
if(ZLIB_FOUND)
    message(STATUS "ZLIB_FOUND")
else()
    message(STATUS "NO")
endif()






add_library(png_static STATIC IMPORTED)

set_target_properties(png_static PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${libpng16_ROOT}/include/libpng16"
    INTERFACE_LINK_LIBRARIES "ZLIB::ZLIB"
    INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "include/libpng16"
)

set_property(TARGET png_static APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)

set_target_properties(png_static PROPERTIES
    IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "C"
    IMPORTED_LOCATION_RELEASE "${libpng16_ROOT}/lib/libpng16.a"
)

add_library(PNG::PNG ALIAS png_static)
            











find_package(SDL2_image 2.8.2 EXACT REQUIRED) # PATHS "P:\\Downloads\\cpp-libraries\\portaudio\\build\\cmake\\portaudio")

if(SDL2_image_FOUND)
    message(STATUS "SDL2_image_FOUND")
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


target_link_libraries(${PROJECT_NAME} PRIVATE common gpt-2 whisper SDL2::SDL2-static SDL2_image::SDL2_image-static audio)
target_link_options(${PROJECT_NAME} PRIVATE "-static")