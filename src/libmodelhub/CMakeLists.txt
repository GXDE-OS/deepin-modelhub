set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(LIB_NAME deepinmodelhub)

FILE(GLOB_RECURSE SRCS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
    "${CMAKE_SOURCE_DIR}/include/*.h"
)

include_directories("${CMAKE_SOURCE_DIR}")

add_library(${LIB_NAME}
    SHARED
    ${SRCS}
)

target_include_directories(${LIB_NAME}
    PRIVATE
   "${CMAKE_SOURCE_DIR}/3rdparty"
)

target_link_libraries(${LIB_NAME} PRIVATE stdc++fs dl)

install(TARGETS ${LIB_NAME} LIBRARY DESTINATION ${LIB_PLATFORM_DIR})
