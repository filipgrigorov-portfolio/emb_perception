cmake_minimum_required(VERSION 3.16)
project(emb_perception C CXX)

find_package(OpenCV REQUIRED COMPONENTS core imgproc)

if (OpenCV_FOUND)
    message("Found OpenCV: " ${OpenCV_VERSION})
    message("OpenCV include dirs: " ${OpenCV_INCLUDE_DIRS})
endif()

find_package(Threads)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

#set(TENSORFLOW_SOURCE_DIR "" CACHE PATH "tensorflow lite project dir")

# TODO(Filip): make the packages external (separate recipe)
set(src 
    src/detection/detection.cpp
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
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${TF_LITE})
