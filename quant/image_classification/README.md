# Image Classification

The script `eval_quant.py` evaluates cumulative bit flips in the [MLPerf Tiny](https://github.com/mlcommons/tiny) benchmark Image Classification. It evalutes the impact on performance (accuracy) of the 8-bit integer-only quantized model.

## Usage

The model is pre-trained and stored in `../models/cifar10_resnet8.tflite`. The training process is described in the [MLPerf Tiny](https://github.com/mlcommons/tiny) repository. We also provide two other models: `cifar10_densenet121.tflite` and `cifar10_resnet50.tflite`.

```bash
$ python eval_quant.py --help

usage: eval_quant.py [-h] --model_path MODEL_PATH [--n_bits N_BITS] [--data_dir DATA_DIR] [--defend] [--random] [--verbose]

Flip N bits in the quantized model and evaluate the accuracy

optional arguments:
  -h, --help            show this help message and exit
  --model_path MODEL_PATH
                        Path to the model (default: None)
  --n_bits N_BITS       Number of bits to flip (default: 1)
  --data_dir DATA_DIR   Path to the data directory (default: ./data/)
  --defend              Enable TMR for the model (default: False)
  --random              Flip bits randomly in the model. If False, flip bits in the high-precision buffers (default: False)
  --verbose             Print integrity violations (default: False)
```