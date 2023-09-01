#!/bin/bash

TARGET_ARCH=riscv64
TOOLCHAIN_PREFIX=riscv64-unknown-linux-gnu-
BENCHMARK_NAME=$1
GIT_REPO_PATH=$(git rev-parse --show-toplevel)
REV_PARSE_PATH=$(echo $GIT_REPO_PATH | sed 's_/_\\/_g')

if [[ ! "$BENCHMARK_NAME" =~ ^(all|clean|keyword_spotting|anomaly_detection|visual_wake_words|image_classification|mnist_lenet|resnet50)$ ]]; then
    echo "Usage: $0 BENCHMARK_NAME"
    echo "Supported benchmarks are: keyword_spotting"
    echo "                          anomaly_detection"
    echo "                          visual_wake_words"
    echo "                          image_classification"
    echo "                          mnist_lenet"
    echo "                          resnet50"
    exit -1
fi

declare -A shortnames=(
    ["anomaly_detection"]="ad"
    ["visual_wake_words"]="vww"
    ["keyword_spotting"]="kws"
    ["image_classification"]="ic"
    ["mnist_lenet"]="lenet"
    ["resnet50"]="ic"
)

function compile() {
    cd tflite-micro
    sed -i "s/REV_PARSE_PATH_PLACEHOLDER/$REV_PARSE_PATH/g" tensorflow/lite/micro/examples/$1/"${shortnames[$1]}"_test.cc
    make -f tensorflow/lite/micro/tools/make/Makefile test_$1\_test TOOLCHAIN_PREFIX=$TOOLCHAIN_PREFIX TARGET_ARCH=$TARGET_ARCH -j96
    cp gen/linux_riscv64_default/bin/$1_test ../$1\_test.bin
}

if [[ "$1" = "clean" ]]; then
    for item in "${!shortnames[@]}"
    do
        rm ${item}_test.bin
    done
    exit 0
fi

if [[ "$BENCHMARK_NAME" = "resnet50" ]]; then
    MODEL_FILE="$GIT_REPO_PATH/tflite-micro/tensorflow/lite/micro/examples/resnet50/ic_model_data.h"
    if [ ! -f $MODEL_FILE ]; then
        pushd "$GIT_REPO_PATH/tflite-micro/tensorflow/lite/micro/models/"      # To circumvent xxd array naming convention.
        xxd -i "resnet50_model.tflite" > $MODEL_FILE
        popd
    fi
fi

if [[ "$BENCHMARK_NAME" = "all" ]]; then
    for item in "${!shortnames[@]}"
    do
        compile $item
    done
else
    compile $BENCHMARK_NAME
fi
