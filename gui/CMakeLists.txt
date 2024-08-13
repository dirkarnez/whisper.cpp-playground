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

