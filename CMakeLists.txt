cmake_minimum_required(VERSION 3.16)
project(emb_perception C CXX)

# libtorch: bin, include, lib, share
# cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch
# cmake --build . --config Release

find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc highgui)

if (OpenCV_FOUND)
    message("Found OpenCV: " ${OpenCV_VERSION})
    message("OpenCV include dirs: " ${OpenCV_INCLUDE_DIRS})
endif()

find_package(Threads)

# Note: There is a linking conflict with opencv (download https://download.pytorch.org/libtorch/cu102/libtorch-cxx11-abi-shared-with-deps-1.10.2%2Bcu102.zip)
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/3rdparty/libtorch/")
find_package(Torch REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#set(TENSORFLOW_SOURCE_DIR "" CACHE PATH "tensorflow lite project dir")

# TODO(Filip): make the packages external (separate recipe)
set(src 
    src/detection/tflite_object_detection.cpp
    src/main.cpp
)

#if (NOT TENSORFLOW_SOURCE_DIR)
#    message(${CMAKE_CURRENT_LIST_DIR})
#    get_filename_component(TENSORFLOW_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../tensorflow_src/" ABSOLUTE)
#endif()

#add_subdirectory("${TENSORFLOW_SOURCE_DIR}/tensorflow/lite" "${CMAKE_CURRENT_BINARY_DIR}/tensorflow-lite" EXCLUDE_FROM_ALL)

include_directories(${CMAKE_SOURCE_DIR}/3rdparty/tf_lite/include/)

add_executable(${PROJECT_NAME} ${src})
target_include_directories(${PROJECT_NAME} PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
    $<INSTALL_INTERFACE:include/>
    ${OpenCV_INCLUDE_DIRS} 
)

# tf_lite could be install'ed globally
find_library(TF_LITE 
    NAMES libtensorflowlite_c tf_lite tensorflowlite libtensorflowlite_c.so
    HINTS "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/tf_lite"
)
message(${TF_LITE})
if (${TF_LITE})
    message("tf_lite_lib found")
endif()

#target_include_directories(${PROJECT_NAME} PRIVATE "${CMAKE_CURRENT_LIST_DIR}/../tensorflow_src")
target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${TF_LITE})
