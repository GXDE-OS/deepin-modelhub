#tar -czvf sensevoice.cpp.tar.gz ./sensevoice.cpp/
if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/sensevoice.cpp")
    message("find sense voice")
else()
    message("decompression sense voice")
    execute_process(
        COMMAND tar -xzvf sensevoice.cpp.tar.gz
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        RESULT_VARIABLE SENSEVOICECPP_UNZIP_RESULT
    )

    if (NOT SENSEVOICECPP_UNZIP_RESULT EQUAL "0")
        message(FATAL_ERROR "tar sensevoice.cpp.tar.gz failed: ${SENSEVOICECPP_UNZIP_RESULT}")
    endif()
endif()

set(SENSEVOICE_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/lib/${CMAKE_LIBRARY_ARCHITECTURE}/deepin-modelhub/backends/sensevoice.cpp")
set(SENSEVOICE_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/../../src/sensevoice.cpp")
file(MAKE_DIRECTORY ${SENSEVOICE_OUTPUT_DIR})

add_library(sensevoice SHARED IMPORTED GLOBAL)

set_property(TARGET sensevoice PROPERTY IMPORTED_LOCATION ${SENSEVOICE_OUTPUT_DIR}/libsense-voice-core.so)

add_subdirectory(sensevoice.cpp-min)
