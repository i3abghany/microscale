# Visual Wake Words

The script `eval_fp32.py` evaluates cumulative bit flips in the [MLPerf Tiny](https://github.com/mlcommons/tiny) benchmark Visual Wake Words. It evalutes the impact on accuracy.

## Usage

The pre-trained model is available under `../models/vww.h5`. The model is trained based on the code from the [MLPerf Tiny](https://github.com/mlcommons/tiny) repository.

The dataset can be downloaded as follows:

```bash
$ wget https://www.silabs.com/public/files/github/machine_learning/benchmarks/datasets/vw_coco2014_96.tar.gz
$ mkdir -p $HOME/tf_data
$ tar -xzvf ./vw_coco2014_96.tar.gz -C $HOME/tf_data
```

```bash
$ python eval_fp32.py --help

usage: eval_fp32.py [-h] [--data_dir DATA_DIR] --model_path MODEL_PATH [--n_bits N_BITS] [--defend] [--exp_only]
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