import os
import sys
import numpy
import random
from sklearn import metrics
import keras_model
import tensorflow as tf
import pickle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils import get_argparser
from model_object import ModelObject


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
        try:
            auc = metrics.roc_auc_score(y_true, y_pred)
        except ValueError as e:
            print("Model output corrupted - ", e)
        performance.append(auc)

    avg_auc = numpy.mean(numpy.array(performance, dtype=float), axis=0)
    return avg_auc


def get_data(data_dir):
    with open(os.path.join(data_dir, "pickled_data.pkl"), "rb") as f:
        all_data = pickle.load(f)

    with open(os.path.join(data_dir, "y_true.pkl"), "rb") as f:
        y_true = pickle.load(f)

    assert len(all_data) == len(y_true)
    return all_data[:1], y_true[:1]


if __name__ == "__main__":
    args = get_argparser().parse_args()
    if not os.path.exists(args.model_path):
        print(f"Model file not found: {args.model_path}")
        sys.exit(-1)

    all_data, y_true = get_data(args.data_dir)
    obj = ModelObject(keras_model.load_model(args.model_path))

    print(f"Performance before flipping: ", run_inference(obj.get_model(), all_data, y_true))
    for i in range(args.n_bits):
        obj.flip_bit(args.exp_only, args.msb_only, args.mantissa_only, args.verbose)
        print(f"Performance after {i+1} bits: ", run_inference(obj.get_model(), all_data, y_true))

        if args.defend:
            obj.clip()
            print(f"Performance with defence: ", run_inference(obj.get_model(), all_data, y_true))
