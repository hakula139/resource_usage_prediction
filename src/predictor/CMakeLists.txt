# ./src/predictor/CMakeLists.txt

aux_source_directory(. DIR_MODEL_SRCS)
add_library(Model STATIC ${DIR_MODEL_SRCS})
target_link_libraries(Model
    PUBLIC
        "${TORCH_LIBRARIES}"
        Common
)
