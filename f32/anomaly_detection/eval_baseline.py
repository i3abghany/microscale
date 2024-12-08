import os
import sys
import numpy
import random
from sklearn import metrics
import keras_model
import tensorflow as tf
import pickle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils import *
from check import clip_in_range


def run_inference(model, all_data, y_trues):
    performance = []
    for i in range(len(all_data)):
        y_true = y_trues[i]
        data = all_data[i]
        y_pred = [0.0 for _ in data]
        for file_idx, datum in enumerate(data):
            pred = model.predict(datum, verbose=False)
            errors = numpy.mean(numpy.square(datum - pred), axis=1)
            y_pred[file_idx] = numpy.mean(errors)

        auc = metrics.roc_auc_score(y_true, y_pred)
        performance.append(auc)

    averaged_performance = numpy.mean(numpy.array(performance, dtype=float), axis=0)
    return averaged_performance


if __name__ == "__main__":
    args = get_argparser().parse_args()
    model_file = "../models/model_ToyCar.hdf5"
    if not os.path.exists(model_file):
        print(f"Model file not found: {model_file}")
        sys.exit(-1)

    model = keras_model.load_model(model_file)
    weights = model.get_weights()
    weight_indices = range(len(weights))
    lengths = []

    for i in weight_indices:
        lengths.append(weights[i].size)

    min_weight, max_weight = get_weight_ranges(model)

    with open(os.path.join(args.data_dir, "pickled_data.pkl"), "rb") as f:
        all_data = pickle.load(f)

    with open(os.path.join(args.data_dir, "y_true.pkl"), "rb") as f:
        y_true = pickle.load(f)

    assert len(all_data) == len(y_true)

    print(f"Performance before flipping: ", run_inference(model, all_data, y_true))
    for i in range(args.n_bits):
        layer_idx = random.choices(weight_indices, weights=lengths, k=1)[0]
        weight_idx = random.randint(0, weights[layer_idx].size - 1)

        if args.exp_only:
            bit_idx = random.randint(22, 32)
        elif args.mantissa_only:
            bit_idx = random.randint(0, 21)
        elif args.msb_only:
            bit_idx = 30
        else:
            bit_idx = random.randint(0, 32)

        print(f"Flipping bit {bit_idx} in layer {layer_idx}, weight {weight_idx}")
        model_flip_bit(model, layer_idx, weight_idx, bit_idx)
        print(f"Performance after {i+1} bits: ", run_inference(model, all_data, y_true))

        if args.defend:
            clip_in_range(model, min_weight, max_weight)
            print(f"Performance with defence: ", run_inference(model, all_data, y_true))

        if not args.cumulative and not args.defend:
            model_flip_bit(model, layer_idx, weight_idx, bit_idx)
