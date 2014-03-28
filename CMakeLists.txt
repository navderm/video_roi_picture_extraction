CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
if(CMAKE_BUILD_TYPE STREQUAL Release)
  SET (PROJECT_NAME videopictureextraction)
else()
  SET(CMAKE_BUILD_TYPE Debug)
  SET (PROJECT_NAME videopictureextractiond)
  SET(CMAKE_CXX_FILES "-g -Wall")
endif()

PROJECT(${PROJECT_NAME})

SET(BUILD_SHARED_LIBS ON)

FIND_PACKAGE(OpenCV REQUIRED)
FIND_PACKAGE(Boost REQUIRED)

SET(SRC src/video_picture_extraction.cpp
        src/main.cpp)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

include_directories(${PROJECT_SOURCE_DIR}/include ${OpenCV_INCLUDE_DIRS})

# add_library(${PROJECT_NAME} ${SRC})
add_executable(${PROJECT_NAME} ${SRC})
# set_target_properties(${PROJECT_NAME} PROPERTIES SOVERSION ${perception_kit_VERSION})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${Boost_LIBS})
