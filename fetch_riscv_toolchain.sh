#!/bin/bash

mkdir -p $HOME/cross
cd $HOME/cross
wget -O $HOME/cross/riscv64.tar.xz https://github.com/riscv-collab/riscv-gnu-toolchain/releases/download/2021.01.26/riscv64-glibc-ubuntu-20.04-nightly-2021.01.26-nightly.tar.gz
tar xvf riscv64.tar.xz
rm riscv64.tar.xz
echo "$PWD/riscv64/bin must be added to PATH manually."

