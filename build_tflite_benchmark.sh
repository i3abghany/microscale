#!/bin/bash

TARGET_ARCH=riscv64
TOOLCHAIN_PREFIX=riscv64-unknown-linux-gnu-
BENCHMARK_NAME=$1
GIT_REPO_PATH=$(git rev-parse --show-toplevel)
REV_PARSE_PATH=$(echo $GIT_REPO_PATH | sed 's_/_\\/_g')

if [[ ! "$BENCHMARK_NAME" =~ ^(keyword_spotting|anomaly_detection|visual_wake_words|image_classification|mnist_lstm)$ ]]; then
    echo "Usage: $0 BENCHMARK_NAME"
    echo "Supported benchmarks are: keyword_spotting"
    echo "                          anomaly_detection"
    echo "                          visual_wake_words"
    echo "                          image_classification"
    echo "                          mnist_lstm"
    exit -1
fi

declare -A shortnames=(
    ["anomaly_detection"]="ad"
    ["visual_wake_words"]="vww"
    ["keyword_spotting"]="kws"
    ["image_classification"]="ic"
    ["mnist_lstm"]="lstm"
)

cd tflite-micro
sed -i "s/REV_PARSE_PATH_PLACEHOLDER/$REV_PARSE_PATH/g" tensorflow/lite/micro/examples/$BENCHMARK_NAME/"${shortnames[$BENCHMARK_NAME]}"_test.cc
make -f tensorflow/lite/micro/tools/make/Makefile test_$BENCHMARK_NAME\_test TOOLCHAIN_PREFIX=$TOOLCHAIN_PREFIX TARGET_ARCH=$TARGET_ARCH -j48
cp gen/linux_riscv64_default/bin/$BENCHMARK_NAME\_test ../$BENCHMARK_NAME\_test
