# Anomaly Detection

The script `eval_fp32.py` evaluates cumulative bit flips in the [MLPerf Tiny](https://github.com/mlcommons/tiny) benchmark Anomaly Detection. It evalutes the impact on performance (AUC-ROC).

## Usage

```bash
$ tar -xzvf ./data/data.tar.gz -C ./data
$ python eval_fp32.py --help

usage: eval_baseline.py [-h] [--data_dir DATA_DIR] [--n_bits N_BITS] [--exp_only EXP_ONLY] [--mantissa_only MANTISSA_ONLY] [--msb_only MSB_ONLY]

Flip N bit in a model and evaluate the performance

optional arguments:
  -h, --help            show this help message and exit
  --data_dir DATA_DIR   Path to the pickled data file (default: ./data/)
  --defend DEFEND       Enable the FP32 model defence (default: True)
  --n_bits N_BITS       Number of bits to flip (default: 1)
  --exp_only EXP_ONLY   Flip only the exponent bits (default: False)
  --mantissa_only MANTISSA_ONLY
                        Flip only the mantissa bits (default: False)
  --msb_only MSB_ONLY   Flip only the MSB of the exponent (default: False)
  --cumulative CUMULATIVE
                        Flip bits cumulatively (default: True)
```