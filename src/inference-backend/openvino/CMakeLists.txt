set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(PLUGIN_NAME backend-openvino)
set(PLUGIN_JSON_FILE libbackend-openvino.json)

FILE(GLOB_RECURSE SRCS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.json"
)

find_package(OpenVINO REQUIRED COMPONENTS Runtime)

add_library(${PLUGIN_NAME}
    SHARED
    ${SRCS}
)

set_target_properties(${PLUGIN_NAME} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ../../)
configure_file(${PLUGIN_JSON_FILE} ${CMAKE_BINARY_DIR}/src/${PLUGIN_JSON_FILE} COPYONLY)

target_link_libraries(${PLUGIN_NAME}
    deepinmodelhub
    openvino::runtime
    )

target_include_directories(${PLUGIN_NAME}
    PRIVATE
    "${CMAKE_SOURCE_DIR}/3rdparty"
    )

install(TARGETS ${PLUGIN_NAME} DESTINATION ${PLUGIN_BACKEND_DIR})
install(FILES ${PLUGIN_JSON_FILE} DESTINATION ${PLUGIN_BACKEND_DIR})
