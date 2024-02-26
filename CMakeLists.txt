cmake_minimum_required(VERSION 3.18)

project("whisper.cpp-playground")

set (THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads)

# find_package(portaudio 19 EXACT REQUIRED) # PATHS "P:\\Downloads\\cpp-libraries\\portaudio\\build\\cmake\\portaudio")

if(Threads_FOUND)
    message(STATUS ${})
else()
    message(STATUS "NO")
endif()

add_executable(${PROJECT_NAME}
    "src/main.cpp"
)

target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_11)

# common common-sdl whisper
target_link_libraries(${PROJECT_NAME} PRIVATE Threads::Threads)
target_link_options(${PROJECT_NAME} PRIVATE "-static")

