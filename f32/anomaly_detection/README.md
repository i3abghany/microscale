# Anomaly Detection

The script `eval_bit_flips.py` evaluates cumulative bit flips in the MLTiny Perf benchmark Anomaly Detection. It evalutes the impact on performance (AUC-ROC).

## Usage

```bash
$ tar -xzvf ./data/data.tar.gz -C ./data
$ python eval_bit_flips.py --help

usage: eval_baseline.py [-h] [--data_dir DATA_DIR] [--n_bits N_BITS] [--exp_only EXP_ONLY] [--mantissa_only MANTISSA_ONLY] [--msb_only MSB_ONLY]

Flip N bit in a model and evaluate the performance

optional arguments:
  -h, --help            show this help message and exit
  --data_dir DATA_DIR   Path to the pickled data file (default: ./data/)
  --n_bits N_BITS       Number of bits to flip (default: 1)
  --exp_only EXP_ONLY   Flip only the exponent bits (default: False)
  --mantissa_only MANTISSA_ONLY
                        Flip only the mantissa bits (default: False)
  --msb_only MSB_ONLY   Flip only the MSB (default: False)
```