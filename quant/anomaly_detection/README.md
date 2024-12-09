# Anomaly Detection

The script `eval_quant.py` evaluates cumulative bit flips in the [MLPerf Tiny](https://github.com/mlcommons/tiny) benchmark Anomaly Detection. It evalutes the impact on performance (AUC-ROC) of the 8-bit integer-only quantized model.

## Usage

The model is pre-trained and stored in `../models/ad.tflite`. The training process is described in the [MLPerf Tiny](https://github.com/mlcommons/tiny) repository.

```bash
$ tar -xzvf ./data/data.tar.gz -C ./data
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