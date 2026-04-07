add_subdirectory(llama.cpp)

if (ENABLE_SENSEVOICE)
    add_subdirectory(sensevoice.cpp)
endif()

find_package(OpenVINO)
if (OpenVINO_FOUND)
    message("found openvino")
    add_subdirectory(openvino)
endif()

