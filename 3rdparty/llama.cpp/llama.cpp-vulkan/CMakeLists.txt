
message("Enbale vulkan llama.cpp")

add_custom_target(llama-vulkan
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/generate.sh ${CMAKE_CURRENT_SOURCE_DIR}/../llama.cpp ${LLAMA_OUTPUT_DIR}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/generate.sh
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

install(FILES ${LLAMA_OUTPUT_DIR}/libllama-vulkan.so DESTINATION ${LLAMA_INSTALL_DIR})
install(FILES ${LLAMA_OUTPUT_DIR}/libggml-vulkan.so DESTINATION ${LLAMA_INSTALL_DIR})
install(FILES ${LLAMA_OUTPUT_DIR}/libggml-base-vulkan.so DESTINATION ${LLAMA_INSTALL_DIR})
install(FILES ${LLAMA_OUTPUT_DIR}/libggml-cpu-vulkan.so DESTINATION ${LLAMA_INSTALL_DIR})

add_dependencies(llama llama-vulkan)
add_dependencies(common llama-vulkan)
