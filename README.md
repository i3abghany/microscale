# microscale

Currently, the Keyword Spotting MLTiny Perf benchmark is ported to the TFLite Micro C++ framework. Compiling and executing a binary that runs inference of the Keyword Spotting dataset inside GEM5 can be done as follows.

```shell
git submodule init
git submodule update

# Downloads a RISC-V toolchain into $HOME/cross
./fetch_riscv_toolchain.sh     
export PATH=$PATH:$HOME/cross/riscv/bin

# Compiles TFLite Micro and the Keyword Spotting benchmark code & links them together
./build_tflite_benchmark.sh    

# Runs the generated binary inside GEM5. Assumes that RISCV gem5.opt is available in $M5_PATH/build
./run_system.sh $M5_PATH       
```
