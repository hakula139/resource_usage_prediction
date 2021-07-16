# ./tests/common/CMakeLists.txt

aux_source_directory(. DIR_TESTS_COMMON)
add_library(TestsCommon STATIC ${DIR_TESTS_COMMON})
target_include_directories(TestsCommon
    INTERFACE
        "${CMAKE_CURRENT_SOURCE_DIR}"
)
