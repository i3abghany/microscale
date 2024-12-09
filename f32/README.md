# Testing F32 Models

This directory contains the test scripts for the F32 models. Included are the following benchmarks:

- Anomaly Detection: AutoEncoder
- Image Classification: ResNet-8

## Usage

Each benchmark has its own test script. The test script can be run with the following command:

```bash
$ python eval_fp32.py --model_path <model_path> [--n_bits <n_bits>] [--defend] [--exp_only] [--mantissa_only] [--msb_only]
```

The arguments are as follows:

- `--model_path`: Path to the model file
- `--n_bits`: Number of bits to flip (default: 1)
- `--defend`: Enable the FP32 model defence (default: False)
- `--exp_only`: Flip only the exponent bits (default: False)
- `--mantissa_only`: Flip only the mantissa bits (default: False)
- `--msb_only`: Flip only the MSB of the exponent (default: False)

The pre-trained models are available under `../models/`. The models are trained based on the code from the [MLPerf Tiny](https://github.com/mlcommons/tiny) repository. The MLPerf Tiny benchmarks code is under the Apache License 2.0. The license can be found in the [MLPERF_TINY_LICENSE](../LICENSES/MLPERF_TINY_LICENSE) file.
