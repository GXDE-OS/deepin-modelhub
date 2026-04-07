
message("Enbale avx2 llama.cpp")

add_custom_target(llama-avx2
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/generate.sh ${CMAKE_CURRENT_SOURCE_DIR}/../llama.cpp ${LLAMA_OUTPUT_DIR}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/generate.sh
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

install(FILES ${LLAMA_OUTPUT_DIR}/libllama-avx2.so DESTINATION ${LLAMA_INSTALL_DIR})
install(FILES ${LLAMA_OUTPUT_DIR}/libggml-avx2.so DESTINATION ${LLAMA_INSTALL_DIR})
install(FILES ${LLAMA_OUTPUT_DIR}/libggml-base-avx2.so DESTINATION ${LLAMA_INSTALL_DIR})
install(FILES ${LLAMA_OUTPUT_DIR}/libggml-cpu-avx2.so DESTINATION ${LLAMA_INSTALL_DIR})

add_dependencies(llama llama-avx2)
add_dependencies(common llama-avx2)
