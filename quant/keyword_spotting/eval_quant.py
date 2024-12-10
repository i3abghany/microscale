import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import tensorflow as tf
import numpy as np
import random

import kws_util
import get_dataset as kws_data
from model_object import ModelObject
from utils import get_argparser

import yaml


def run_inference(model_path, ds_test):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_scale, input_zero_point = input_details[0]["quantization"]
    output_data = []
    labels = []
    eval_data = ds_test.unbatch().batch(1).as_numpy_iterator()
    for dat, label in eval_data:
        dat_q = np.array(dat / input_scale + input_zero_point, dtype=np.int8)
        interpreter.set_tensor(input_details[0]["index"], dat_q)
        interpreter.invoke()
        output_data.append(np.argmax(interpreter.get_tensor(output_details[0]["index"])))
        labels.append(label[0])

    num_correct_predictions = np.sum(np.array(labels) == output_data)
    acc = num_correct_predictions / len(labels)
    return acc


if __name__ == "__main__":
    yaml_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    ds_train, ds_test, ds_val = kws_data.get_training_data(config)
    args = get_argparser().parse_args()

    model_obj = ModelObject(args.model_path, defended=args.defend)

    baseline_acc = run_inference(args.model_path, ds_test)
    print(f"Baseline Accuracy: {baseline_acc}")

    for _ in range(args.n_bits):
        model_obj.flip_bit(rand=args.random)

    if args.defend:
        model_obj.check_integrity(verbose=args.verbose)

    modified_model_path = "./quant_model_modified.tflite"
    model_obj.write_model(modified_model_path)

    acc = run_inference(modified_model_path, ds_test)
    print(f"Attacked model ({args.defend=}) Accuracy: {acc}")

    os.remove(modified_model_path)
