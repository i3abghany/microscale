# Keyword Spotting

The script `eval_fp32.py` evaluates cumulative bit flips in the [MLPerf Tiny](https://github.com/mlcommons/tiny) benchmark Anomaly Detection. It evalutes the impact on performance (AUC-ROC).

## Usage

The pre-trained model is available under `../models/kws_ds_cnn.keras`. The model is trained based on the code from the [MLPerf Tiny](https://github.com/mlcommons/tiny) repository.

The dataset is automatically downloaded and stored in the `$HOME/tf_data` directory.

```bash
$ python eval_fp32.py --help

usage: eval_fp32.py [-h] [--data_dir DATA_DIR] --model_path MODEL_PATH [--n_bits N_BITS] [--defend] [--exp_only] [--mantissa_only] [--msb_only] [--verbose]

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
  --verbose             Print all the bits being flipped (default: False)
```