cmake_minimum_required(VERSION 3.14)
project(AudioGuard VERSION 1.0.0 LANGUAGES CXX)

##Global Settings

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

#Dependency Management
include(FetchContent)

#PyBind11
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11
    GIT_TAG        v2.10.0
)

FetchContent_MakeAvailable(pybind11)


#FFMPEG
#find_package(PkgConfig REQUIRED)
#pkg_check_modules(LIBAV REQUIRED IMPORTED_TARGET
#libavformat libavcodec libswresample libavutil)


#ONXX 
#find_package(OnnxRuntime REQUIRED)

#Sources
set(CORE_SOURCES
    src/Preprocessor.cpp 
    src/AudioLoader.cpp
    src/InferenceEngine.cpp)

#Targets:Python Module
pybind11_add_module(audioguard_core
    bindings/python_bindings.cpp
    ${CORE_SOURCES}    
)

#Defining .h location
target_include_directories(audioguard_core PRIVATE include)
target_link_libraries(audioguard_core PRIVATE PkgConfig::LIBAV)

#Target: Executable
add_executable(AudioGuardApp
        app/main.cpp
        ${CORE_SOURCES}
)

target_include_directories(AudioGuardApp PRIVATE include)
target_link_libraries(AudioGuardApp PRIVATE PkgConfig::LIBAV)
