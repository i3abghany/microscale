M5_PATH=$1
BENCHMARK_EXECUTABLE=$2

if [ "$M5_PATH" = "" ] || [ "$BENCHMARK_EXECUTABLE" = "" ]
then
    echo "usage: $0 M5_PATH BENCHMARK_EXECUTABLE"
    exit -1
fi

$M5_PATH/build/RISCV/gem5.fast                   \
    --outdir=out/m5out_$BENCHMARK_EXECUTABLE     \
    $M5_PATH/configs/example/se.py               \
    --cmd=./$BENCHMARK_EXECUTABLE                \
    --warmup-insts=100000000                     \
    --sys-clock=1GHz                             \
    --cpu-clock=1GHz                             \
    --cpu-type=MinorCPU                          \
    --param='system.cpu[0].decodeInputWidth=1'   \
    --param='system.cpu[0].executeInputWidth=1'  \
    --param='system.cpu[0].executeIssueLimit=1'  \
    --caches                                     \
    --l1d_size=32kB                              \
    --l1i_size=32kB                              \
    --l1d_assoc=2                                \
    --l1i_assoc=2                                \
    --cacheline_size=32                          \
    --mem-type=LPDDR3_1600_1x32                  \
    --mem-size=256MB                             \
    --bp-type=BiModeBP                           &
