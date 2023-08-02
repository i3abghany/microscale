#!/bin/bash

TARGET_ARCH=riscv64
TOOLCHAIN_PREFIX=riscv64-unknown-linux-gnu-
BENCHMARK_NAME=$1
if test -z $BENCHMARK_NAME
then
    echo "Usage: $0 BENCHMARK_NAME"
    echo "Benchmarks supported are: keyword_spotting"
    echo "                          anomaly_detection"
    echo "                          visual_wake_words"
    exit -1
fi

cd tflite-micro
make -f tensorflow/lite/micro/tools/make/Makefile test_$BENCHMARK_NAME\_test TOOLCHAIN_PREFIX=$TOOLCHAIN_PREFIX TARGET_ARCH=$TARGET_ARCH -j48
cp gen/linux_riscv64_default/bin/$BENCHMARK_NAME\_test ../$BENCHMARK_NAME\_test
