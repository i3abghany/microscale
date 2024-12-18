# microscale

Currently, the Keyword Spotting, Anomaly Detection, Visual Wake Words, and Image Classification [MLPerf Tiny](https://github.com/mlcommons/tiny) benchmarks are ported to the TFLite Micro C++ framework. Other image classification tasks are also ported (LeNet-MNIST, ResNet50-CIFAR10, and DenseNet-121-CIFAR10). Compiling and executing a binary that runs inference using one of the benchmarks using the baseline system configuration inside GEM5 can be done as follows.

```shell

# Initialize the tflite-micro & tensorflow fork submodules
git submodule init
git submodule update

# Downloads a RISC-V toolchain into $HOME/cross
./fetch_riscv_toolchain.sh     
export PATH=$PATH:$HOME/cross/riscv/bin

# Compiles TFLite Micro & the Keyword Spotting benchmark to the current directory
./build_tflite_benchmark.sh keyword_spotting

# Runs the generated binary inside GEM5. Assumes that RISCV gem5.opt is available in $M5_PATH/build
./run_system.sh $M5_PATH keyword_spotting_test
```
