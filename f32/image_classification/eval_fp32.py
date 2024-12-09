import os
import sys
import numpy as np
import random
from sklearn import metrics
import keras_model
import tensorflow as tf
import pickle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils import get_argparser, model_flip_bit, get_weight_ranges
from check import clip_in_range


def load_cifar10():
    (_, _), (test_imgs, test_labels) = tf.keras.datasets.cifar10.load_data()
    test_labels = tf.keras.utils.to_categorical(test_labels, 10)

    # Randomly sample 1000 images
    np.random.seed(112233)
    sample_indices = np.random.choice(len(test_imgs), 1000)
    test_imgs = test_imgs[sample_indices]
    test_labels = test_labels[sample_indices]

    return test_imgs, test_labels


if __name__ == "__main__":

    test_data, test_labels = load_cifar10()

    args = get_argparser().parse_args()
    model_file = "../models/resnet8_cifar10.h5"
    if not os.path.exists(model_file):
        print(f"Model file not found: {model_file}")
        sys.exit(-1)

    model = tf.keras.models.load_model(model_file)
    weights = model.get_weights()
    weight_indices = range(len(weights))
    lengths = []

    for i in weight_indices:
        lengths.append(weights[i].size)

    min_weight, max_weight = get_weight_ranges(model)

    acc = model.evaluate(x=test_data, y=test_labels, batch_size=64, verbose=1, return_dict=True)["accuracy"]
    print(f"Accuracy before flipping: ", acc)
    for i in range(args.n_bits):
        layer_idx = random.choices(weight_indices, weights=lengths, k=1)[0]
        weight_idx = random.randint(0, weights[layer_idx].size - 1)

        if args.exp_only:
            bit_idx = random.randint(22, 31)
        elif args.mantissa_only:
            bit_idx = random.randint(0, 21)
        elif args.msb_only:
            bit_idx = 30
        else:
            bit_idx = random.randint(0, 31)

        print(f"Flipping bit {bit_idx} in layer {layer_idx}, weight {weight_idx}")
        model_flip_bit(model, layer_idx, weight_idx, bit_idx)

        acc = model.evaluate(x=test_data, y=test_labels, batch_size=64, verbose=0, return_dict=True)["accuracy"]
        print(f"Accuracy after {i+1} bits: ", acc)

        if args.defend:
            clip_in_range(model, min_weight, max_weight)
            acc = model.evaluate(x=test_data, y=test_labels, batch_size=64, verbose=0, return_dict=True)["accuracy"]
            print(f"Accuracy with defence: ", acc)
