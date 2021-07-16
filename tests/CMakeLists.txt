# ./tests/CMakeLists.txt

add_subdirectory(common)

aux_source_directory(. DIR_TESTS)
add_executable(generator ${DIR_TESTS})
target_link_libraries(generator
    PUBLIC
        TestsCommon
)
