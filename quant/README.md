# Testing Quantized Models

This directory contains the test scripts for the INT8-quantized models. Included are the following benchmarks:

| Benchmark | Model(s) | Dataset | Description |
| :--- | :--- | :--- | :--- |
| [Keyword Spotting](./keyword_spotting) | DS_CNN | [Speech Commands](https://www.tensorflow.org/datasets/catalog/speech_commands) | Detects a keyword in a short audio clip |
| [Image Classification](./image_classification) | ResNet-8, ResNet-50, DenseNet-121 | [CIFAR-10](https://www.tensorflow.org/datasets/catalog/cifar10) | Classifies images into 10 classes |
| [Visual Wake Words](./visual_wake_words) | MobileNetV1 | [MS-COCO](https://arxiv.org/abs/1405.0312v3) | Detects person presence in an image |
| [Anomaly Detection](./anomaly_detection) | AutoEncoder | [ToyADMOS](https://arxiv.org/abs/1908.03299) | Detects anomalous machines using machine noise data |

## Usage

Each benchmark has its own test script, but we provide a top-level script that calls the individual test scripts. The top-level script is `main.py`. The script takes the following arguments:

```bash
$ python main.py --help

usage: main.py [-h] --model_path MODEL_PATH [--n_bits N_BITS] [--data_dir DATA_DIR] [--defend] [--random] [--verbose] --benchmark
               {keyword_spotting,image_classification,visual_wake_words,anomaly_detection,kws,ic,vww,ad}

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
  --benchmark {keyword_spotting,image_classification,visual_wake_words,anomaly_detection,kws,ic,vww,ad}
                        The benchmark to run. Supports either full name (e.g. keyword_spotting) or abbreviation (e.g. kws) (default: None)
```

The pre-trained models are available under `./models/`. The models are trained based on the code from the [MLPerf Tiny](https://github.com/mlcommons/tiny) repository. The MLPerf Tiny benchmarks code is under the Apache License 2.0. The license can be found in the [MLPERF_TINY_LICENSE](../LICENSES/MLPERF_TINY_LICENSE.md) file.
