# ./src/CMakeLists.txt

add_subdirectory(model)

aux_source_directory(. DIR_SRCS)
add_executable(predictor ${DIR_SRCS})
target_link_libraries(predictor
    PUBLIC
        "${TORCH_LIBRARIES}"
        Model
)
