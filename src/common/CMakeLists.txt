# ./src/common/CMakeLists.txt

aux_source_directory(. DIR_SRC_COMMON)
add_library(Common STATIC ${DIR_SRC_COMMON})
target_include_directories(Common
    INTERFACE
        "${CMAKE_CURRENT_SOURCE_DIR}"
)
