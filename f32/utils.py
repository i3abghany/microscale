import numpy as np
import argparse


def float_toggle_bit(f, b):
    if isinstance(f, np.ndarray):
        assert f.dtype == np.float32, "Array must be of type np.float32"

    as_int = f.view(np.uint32)
    as_int = as_int ^ np.uint32(1 << b)
    f = as_int.view(np.float32)
    return f


def float_reset_bit(f, b):
    if isinstance(f, np.ndarray):
        assert f.dtype == np.float32, "Array must be of type np.float32"

    as_int = f.view(np.uint32)
    as_int = as_int & np.uint32(~(1 << b))
    f = as_int.view(np.float32)
    return f


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
        "--model_path",
        type=str,
        help="Path to the model file",
        required=True,
    )

    parser.add_argument(
        "--n_bits",
        type=int,
        default=1,
        help="Number of bits to flip",
    )

    parser.add_argument(
        "--defend",
        default=False,
        action="store_true",
        help="Enable the FP32 model defence",
    )

    parser.add_argument(
        "--exp_only",
        default=False,
        action="store_true",
        help="Flip only the exponent bits",
    )

    parser.add_argument(
        "--mantissa_only",
        default=False,
        action="store_true",
        help="Flip only the mantissa bits",
    )

    parser.add_argument(
        "--msb_only",
        default=False,
        action="store_true",
        help="Flip only the MSB of the exponent",
    )

    return parser
