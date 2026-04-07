
set(BUILD_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/libllama.so)

message("Enbale min llama.cpp")

add_custom_target(llama-min
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/generate.sh ${CMAKE_CURRENT_SOURCE_DIR}/../llama.cpp ${LLAMA_OUTPUT_DIR}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/generate.sh
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

install(FILES ${LLAMA_OUTPUT_DIR}/libllama.so DESTINATION ${LLAMA_INSTALL_DIR})
install(FILES ${LLAMA_OUTPUT_DIR}/libggml.so DESTINATION ${LLAMA_INSTALL_DIR})
install(FILES ${LLAMA_OUTPUT_DIR}/libggml-base.so DESTINATION ${LLAMA_INSTALL_DIR})
install(FILES ${LLAMA_OUTPUT_DIR}/libggml-cpu.so DESTINATION ${LLAMA_INSTALL_DIR})

add_dependencies(llama llama-min)
add_dependencies(common llama-min)
