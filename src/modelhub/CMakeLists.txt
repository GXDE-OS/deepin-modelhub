
set(CMAKE_INCLUDE_CURRENT_DIR ON)

FILE(GLOB_RECURSE SRCS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)

add_executable(${EXE_NAME}
    ${SRCS}
)

target_include_directories(${EXE_NAME}
    PRIVATE
   "${CMAKE_CURRENT_SOURCE_DIR}/../libmodelhub"
   "${CMAKE_SOURCE_DIR}/3rdparty"
   "${CMAKE_SOURCE_DIR}/3rdparty/llama.cpp"
)

target_link_libraries(${EXE_NAME}
    deepinmodelhub
    pthread
    uuid
    )

install(TARGETS ${EXE_NAME} DESTINATION bin)
