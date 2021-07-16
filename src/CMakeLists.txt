# ./src/CMakeLists.txt

add_subdirectory(common)
add_subdirectory(predictor)

aux_source_directory(. DIR_SRC)
add_executable(predictor ${DIR_SRC})
target_link_libraries(predictor
    PUBLIC
        "${TORCH_LIBRARIES}"
        sciplot::sciplot
        Common
        Model
)
