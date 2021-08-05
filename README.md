# resource_usage_prediction

A resource usage predictor, developed with PyTorch (C++ frontend), written in modern C++.

## Table of Contents

- [resource_usage_prediction](#resource_usage_prediction)
  - [Table of Contents](#table-of-contents)
  - [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
    - [Usage](#usage)

## Getting Started

### Prerequisites

Before installation, you need to have the following dependencies installed.

- [CMake](https://cmake.org/download) 3.5 or later
- [Gnuplot](http://www.gnuplot.info)

### Installation

Execute `./scripts/build.sh` to build the project using CMake.

### Usage

Execute `./bin/generator` to generate a testing dataset.

Execute `./bin/predictor` to start predicting on the testing dataset.
