# Anomaly Detection

The script `eval_quant.py` evaluates cumulative bit flips in the [MLPerf Tiny](https://github.com/mlcommons/tiny) benchmark Anomaly Detection. It evalutes the impact on performance (AUC-ROC) of the 8-bit integer-only quantized model.

## Usage

```bash
$ tar -xzvf ./data/data.tar.gz -C ./data
$ python eval_quant.py --help

Flip N bits in the quantized model and evaluate the accuracy

optional arguments:
  -h, --help            show this help message and exit
  --data_dir DATA_DIR   Path to the data directory (default: ./data/)
  --defend              Enable TMR for the model (default: False)
  --n_bits N_BITS       Number of bits to flip (default: 1)
  --model_path MODEL_PATH
                        Path to the model (default: ../models/ad.tflite)
  --random              Flip bits randomly in the model. If False, flip bits in the high-precision buffers (default: False)
  --verbose             Print integrity violations (default: False)

```