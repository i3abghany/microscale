import copy
import argparse


def get_high_precision_buffers(model):
    res = [], []
    buffers = []
    for i in range(len(model.subgraphs)):
        subg = model.subgraphs[i]
        for ten in subg.tensors:
            INT32_TENSOR_TYPE = 2
            if ten.type == INT32_TENSOR_TYPE:
                if len(model.buffers[ten.buffer].data) <= 32:
                    continue
                res[0].append(ten.buffer)
                res[1].append(len(model.buffers[ten.buffer].data))
                buffers.append(model.buffers[ten.buffer])
    return res


def get_non_null_buffers(model):
    res = [], []
    for i in range(len(model.buffers)):
        if model.buffers[i].data is not None:
            res[0].append(i)
            res[1].append(len(model.buffers[i].data))
    return res


def get_replicas(model_obj):
    r1 = copy.deepcopy(model_obj.buffers)
    r2 = copy.deepcopy(model_obj.buffers)

    return r1, r2


def get_argparser():
    parser = argparse.ArgumentParser(
        description="Flip N bits in the quantized model and evaluate the accuracy",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/",
        help="Path to the data directory",
    )

    parser.add_argument(
        "--defend",
        default=False,
        action="store_true",
        help="Enable TMR for the model",
    )

    parser.add_argument(
        "--n_bits",
        type=int,
        default=1,
        help="Number of bits to flip",
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default="../models/ad.tflite",
        help="Path to the model",
    )

    parser.add_argument(
        "--random",
        default=False,
        action="store_true",
        help="Flip bits randomly in the model. If False, flip bits in the high-precision buffers",
    )

    parser.add_argument(
        "--verbose",
        default=False,
        action="store_true",
        help="Print integrity violations",
    )

    return parser
