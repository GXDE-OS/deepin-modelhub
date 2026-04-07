
set(BUILD_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/libllama.so)

message("Enbale cuda llama.cpp")

add_custom_target(llama-cuda
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/generate.sh ${CMAKE_CURRENT_SOURCE_DIR}/../llama.cpp ${LLAMA_OUTPUT_DIR}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/generate.sh
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

install(FILES ${LLAMA_OUTPUT_DIR}/libllama-cuda.so DESTINATION ${LLAMA_INSTALL_DIR})
install(FILES ${LLAMA_OUTPUT_DIR}/libggml-cuda-cuda.so DESTINATION ${LLAMA_INSTALL_DIR})
install(FILES ${LLAMA_OUTPUT_DIR}/libggml-cuda.so DESTINATION ${LLAMA_INSTALL_DIR})
install(FILES ${LLAMA_OUTPUT_DIR}/libggml-base-cuda.so DESTINATION ${LLAMA_INSTALL_DIR})
install(FILES ${LLAMA_OUTPUT_DIR}/libggml-cpu-cuda.so DESTINATION ${LLAMA_INSTALL_DIR})

add_dependencies(llama llama-cuda)
add_dependencies(common llama-cuda)
