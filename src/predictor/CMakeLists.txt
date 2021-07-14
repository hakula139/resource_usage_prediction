# ./src/predictor/CMakeLists.txt

aux_source_directory(. DIR_MODEL_SRCS)
add_library(Predictor STATIC ${DIR_MODEL_SRCS})
target_link_libraries(Predictor
    PUBLIC
        "${TORCH_LIBRARIES}"
        Common
)
