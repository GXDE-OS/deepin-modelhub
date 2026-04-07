set(BUILD_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/libsense-voice-core.so)

message("Enbale min sensevoice.cpp")

add_custom_target(sensevoice-min
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/generate.sh ${CMAKE_CURRENT_SOURCE_DIR}/../sensevoice.cpp ${SENSEVOICE_OUTPUT_DIR}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/generate.sh
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

install(FILES ${SENSEVOICE_OUTPUT_DIR}/libsense-voice-core.so DESTINATION ${SENSEVOICE_INSTALL_DIR})
# 复用llama.cpp的ggml
#install(FILES ${SENSEVOICE_OUTPUT_DIR}/libggml.so DESTINATION ${SENSEVOICE_INSTALL_DIR})
#install(FILES ${SENSEVOICE_OUTPUT_DIR}/libggml-base.so DESTINATION ${SENSEVOICE_INSTALL_DIR})
#install(FILES ${SENSEVOICE_OUTPUT_DIR}/libggml-cpu.so DESTINATION ${SENSEVOICE_INSTALL_DIR})

add_dependencies(sensevoice sensevoice-min)
