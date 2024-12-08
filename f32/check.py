import tensorflow as tf
import numpy as np

from utils import float_reset_bit


def clip_in_range(model, min_weight, max_weight):
    weights = model.get_weights()
    for i, var in enumerate(weights):
        w = var.flatten()
        out_of_range_indices = np.where(w < min_weight)[0]
        out_of_range_indices = np.append(out_of_range_indices, np.where(w > max_weight)[0])

        if len(out_of_range_indices) > 0:
            print("Found out-of-range weights: ", w[out_of_range_indices])
            clipped = float_reset_bit(w[out_of_range_indices], 30)
            w[out_of_range_indices] = clipped
            print("Clipping into:              ", clipped)

        weights[i] = w.reshape(var.shape)

    model.set_weights(weights)
