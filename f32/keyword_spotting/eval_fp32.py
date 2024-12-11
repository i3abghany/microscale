import numpy as np
import sys, os

from tensorflow import keras

import get_dataset as kws_data

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from utils import get_argparser
from model_object import ModelObject

import yaml

num_classes = 12  # should probably draw this directly from the dataset.

if __name__ == "__main__":
    args = get_argparser().parse_args()
    yaml_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    args = get_argparser().parse_args()
    if not os.path.exists(args.model_path):
        print(f"Model file not found: {args.model_path}")
        sys.exit(-1)

    obj = ModelObject(keras.models.load_model(args.model_path))
    model = obj.get_model()

    ds_train, ds_test, ds_val = kws_data.get_training_data(config)
    acc = model.evaluate(ds_test, batch_size=64, verbose=0, return_dict=True)
    print(f"Performance before flipping: ", acc["sparse_categorical_accuracy"])
    for i in range(args.n_bits):
        obj.flip_bit(args.exp_only, args.msb_only, args.mantissa_only, args.verbose)
        acc = model.evaluate(ds_test, batch_size=64, verbose=0, return_dict=True)
        print(f"Performance after {i+1} bits: ", acc["sparse_categorical_accuracy"])

        if args.defend:
            obj.clip()
            acc = model.evaluate(ds_test, batch_size=64, verbose=0, return_dict=True)
            print(f"Performance with defence: ", acc["sparse_categorical_accuracy"])
