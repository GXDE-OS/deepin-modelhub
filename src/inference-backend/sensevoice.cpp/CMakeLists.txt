set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(PLUGIN_NAME backend-sensevoicecpp)
set(PLUGIN_JSON_FILE libbackend-sensevoicecpp.json)

FILE(GLOB_RECURSE SRCS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.json"
)

add_library(${PLUGIN_NAME}
    SHARED
    ${SRCS}
)

set_target_properties(${PLUGIN_NAME} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ../../)
configure_file(${PLUGIN_JSON_FILE} ${CMAKE_BINARY_DIR}/src/${PLUGIN_JSON_FILE} COPYONLY)

target_include_directories(${PLUGIN_NAME}
    PRIVATE
    "${CMAKE_SOURCE_DIR}/3rdparty/sensevoice.cpp/sensevoice.cpp/sense-voice/csrc"
    "${CMAKE_SOURCE_DIR}/3rdparty/sensevoice.cpp/sensevoice.cpp/sense-voice/csrc/third-party/ggml/include"
    "${CMAKE_SOURCE_DIR}/3rdparty"
    )

target_link_libraries(${PLUGIN_NAME}
    sensevoice
    )

install(TARGETS ${PLUGIN_NAME} DESTINATION ${PLUGIN_BACKEND_DIR})
install(FILES ${PLUGIN_JSON_FILE} DESTINATION ${PLUGIN_BACKEND_DIR})
