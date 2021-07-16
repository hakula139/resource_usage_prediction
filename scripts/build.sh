#!/bin/bash

# Install sciplot

SCIPLOT_PATH=/usr/local/include/sciplot/sciplot.hpp
if [ ! -f "$SCIPLOT_PATH" ]; then
  SRC_DIR=$PWD
  cd lib/sciplot
  mkdir -p build && \
    cd build && \
    cmake .. && \
    cmake --build . --target install
  cd $SRC_DIR
fi

# Install main project

mkdir -p build && \
  cd build && \
  cmake .. && \
  cmake --build .
