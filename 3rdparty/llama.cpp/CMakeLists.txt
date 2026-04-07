#tar -czvf llama.cpp.tar.gz ./llama.cpp
#pre: af148c9386da825a60c7038549c121c35ca56b50
#cur: 553f1e46e9e864514bbd6bf4009146db66be0541
if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp")
    message("find llama.cpp")
else()
    message("decompression llama.cpp")
    execute_process(
        COMMAND tar -xzvf llama.cpp.tar.gz
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        RESULT_VARIABLE LLAMACPP_UNZIP_RESULT
    )

    if (NOT LLAMACPP_UNZIP_RESULT EQUAL "0")
        message(FATAL_ERROR "tar llama.cpp.tar.gz failed: ${LLAMACPP_UNZIP_RESULT}")
    endif()
endif()

set(LLAMA_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/lib/${CMAKE_LIBRARY_ARCHITECTURE}/deepin-modelhub/backends/llama.cpp")
set(LLAMA_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/../../src/llama.cpp")
file(MAKE_DIRECTORY ${LLAMA_OUTPUT_DIR})

add_library(llama SHARED IMPORTED GLOBAL)
add_library(common STATIC IMPORTED GLOBAL)

set_property(TARGET llama PROPERTY IMPORTED_LOCATION ${LLAMA_OUTPUT_DIR}/libllama.so)
set_property(TARGET common PROPERTY IMPORTED_LOCATION ${LLAMA_OUTPUT_DIR}/libcommon.a)

add_subdirectory(llama.cpp-min)

if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "x86_64")
    add_subdirectory(llama.cpp-avx2)

    find_package(CUDAToolkit)
    if (CUDAToolkit_FOUND)
        add_subdirectory(llama.cpp-cuda)
    endif()
endif()
