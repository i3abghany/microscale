M5_PATH=$1

if [[ "$M5_PATH" = "" ]]
then
    echo "usage: $0 M5_PATH"
    exit -1
fi

$M5_PATH/build/RISCV/gem5.opt                    \
    $M5_PATH/configs/example/se.py               \
    --cmd=./kws_test_riscv                       \
    --sys-clock=225MHz                           \
    --cpu-clock=225MHz                           \
    --cpu-type=MinorCPU                          \
    --param='system.cpu[0].decodeInputWidth=1'   \
    --param='system.cpu[0].executeInputWidth=1'  \
    --param='system.cpu[0].executeIssueLimit=1'  \
    --caches                                     \
    --l1d_size=32kB                              \
    --l1i_size=32kB                              \
    --l1d_assoc=8                                \
    --l1i_assoc=8                                \
    --l1d-hwp-type=StridePrefetcher              \
    --cacheline_size=64                          \
    --mem-type=LPDDR3_1600_1x32                  \
    --mem-size=256MB                             \
    --bp-type=BiModeBP                           \