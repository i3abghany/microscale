#!/bin/bash

TARGET_ARCH=riscv64
TOOLCHAIN_PREFIX=riscv64-unknown-linux-gnu-

cd tflite-micro
make -f tensorflow/lite/micro/tools/make/Makefile test_keyword_spotting_test TOOLCHAIN_PREFIX=$TOOLCHAIN_PREFIX TARGET_ARCH=$TARGET_ARCH -j48
cp gen/linux_riscv64_default/bin/keyword_spotting_test ../kws_test_riscv
