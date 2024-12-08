import numpy as np
import argparse


def float_toggle_bit(f, b):
    modified_weight = f
    as_int = modified_weight.view(np.uint32)
    as_int = as_int ^ np.uint32(1 << b)
    modified_weight = as_int.view(np.float32)
    return modified_weight


def model_flip_bit(model, layer_idx, flat_weight_idx, bit_idx):
    weights = model.get_weights()
    weight = weights[layer_idx].flat[flat_weight_idx]
    weight = float_toggle_bit(weight, bit_idx)
    weights[layer_idx].flat[flat_weight_idx] = weight
    model.set_weights(weights)


def get_weight_ranges(model):
    weights = model.get_weights()
    min_weight = float("inf")
    max_weight = -float("inf")
    for var in weights:
        min_weight = min(min_weight, np.min(var.flatten()))
        max_weight = max(max_weight, np.max(var.flatten()))
    return min_weight, max_weight


def get_argparser():
    parser = argparse.ArgumentParser(
        description="Flip N bit in a model and evaluate the performance",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/",
        help="Path to the pickled data file",
    )

    parser.add_argument(
        "--n_bits",
        type=int,
        default=1,
        help="Number of bits to flip",
    )

    parser.add_argument(
        "--exp_only",
        type=bool,
        default=False,
        help="Flip only the exponent bits",
    )

    parser.add_argument(
        "--mantissa_only",
        type=bool,
        default=False,
        help="Flip only the mantissa bits",
    )

    parser.add_argument(
        "--msb_only",
        type=bool,
        default=False,
        help="Flip only the MSB of the exponent",
    )

    parser.add_argument(
        "--cumulative",
        type=bool,
        default=True,
        help="Flip bits cumulatively",
    )

    return parser
