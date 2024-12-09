# Image Classification

The script `eval_fp32.py` evaluates cumulative bit flips in the [MLPerf Tiny](https://github.com/mlcommons/tiny) benchmark Image Classification. It evalutes the impact on performance (accuracy).

## Usage

The model is pre-trained and stored in `../models/resnet8_cifar10.h5` (ResNet-8). The training process is based on the [MLPerf Tiny](https://github.com/mlcommons/tiny) repository.

```bash
$ python eval_fp32.py --help

usage: eval_fp32.py [-h] [--data_dir DATA_DIR] [--model_path MODEL_PATH] [--n_bits N_BITS] [--defend] [--exp_only]
                    [--mantissa_only] [--msb_only]

Flip N bit in a model and evaluate the performance

optional arguments:
  -h, --help            show this help message and exit
  --data_dir DATA_DIR   Path to the pickled data file (default: ./data/)
  --model_path MODEL_PATH
                        Path to the model file (default: None)
  --n_bits N_BITS       Number of bits to flip (default: 1)
  --defend              Enable the FP32 model defence (default: False)
  --exp_only            Flip only the exponent bits (default: False)
  --mantissa_only       Flip only the mantissa bits (default: False)
  --msb_only            Flip only the MSB of the exponent (default: False)

```