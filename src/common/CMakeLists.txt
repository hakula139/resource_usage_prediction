# ./src/common/CMakeLists.txt

aux_source_directory(. DIR_COMMON_SRCS)
add_library(Common STATIC ${DIR_COMMON_SRCS})
target_include_directories(Common
    INTERFACE
        "${CMAKE_CURRENT_SOURCE_DIR}"
)
