# ./src/predictor/CMakeLists.txt

aux_source_directory(. DIR_SRC_PREDICTOR)
add_library(Model STATIC ${DIR_SRC_PREDICTOR})
target_link_libraries(Model
    PUBLIC
        "${TORCH_LIBRARIES}"
        Common
)
