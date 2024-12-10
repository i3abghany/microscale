# Visual Wake Words

The script `eval_quant.py` evaluates cumulative bit flips in the [MLPerf Tiny](https://github.com/mlcommons/tiny) benchmark Visual Wake Words. It evalutes the impact on accuracy of the 8-bit integer-only quantized model.

The dataset can be downloaded as follows:

```bash
$ wget https://www.silabs.com/public/files/github/machine_learning/benchmarks/datasets/vw_coco2014_96.tar.gz
$ mkdir data
$ tar -xzvf ./vw_coco2014_96.tar.gz -C ./data
```

The model is pre-trained and stored in `../models/vww.tflite` (MobileNet-V1). The training process is described in the [MLPerf Tiny](https://github.com/mlcommons/tiny) repository.

```bash

