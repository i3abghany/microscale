import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy
from sklearn import metrics
import tensorflow as tf
import pickle
from model_object import ModelObject
from utils import get_argparser


def run_inference(y_true, data, model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    performance = []
    for i in range(len(all_data)):
        y_true = y_trues[i]
        data = all_data[i]
        y_pred = [0.0 for _ in data]
        for file_idx, datum in enumerate(data):
            output = run_inference_one_sample(interpreter, datum)
            errors = numpy.mean(numpy.square(datum - output), axis=1)
            y_pred[file_idx] = numpy.mean(errors)

        auc = metrics.roc_auc_score(y_true, y_pred)
        performance.append(auc)

    return numpy.mean(numpy.array(performance, dtype=float))


def run_inference_one_sample(interpreter, data):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_data = numpy.array(data, dtype=numpy.float32)
    output_data = numpy.empty_like(data)

    for i in range(input_data.shape[0]):
        interpreter.set_tensor(input_details[0]["index"], input_data[i : i + 1, :])
        interpreter.invoke()
        output_data[i : i + 1, :] = interpreter.get_tensor(output_details[0]["index"])

    return output_data


def get_data(data_dir):
    with open(os.path.join(data_dir, "pickled_data.pkl"), "rb") as fx:
        all_data = pickle.load(fx)

    with open(os.path.join(data_dir, "y_true.pkl"), "rb") as fx:
        y_trues = pickle.load(fx)

    return y_trues, all_data


if __name__ == "__main__":
    args = get_argparser().parse_args()

    model_obj = ModelObject(args.model_path, defended=args.defend)

    y_trues, all_data = get_data(args.data_dir)
    baseline_acc = run_inference(y_trues, all_data, args.model_path)
    print(f"Baseline AUC: {baseline_acc}")

    for _ in range(args.n_bits):
        model_obj.flip_bit(rand=args.random)

    if args.defend:
        model_obj.check_integrity(verbose=args.verbose)

    modified_model_path = "./quant_model_modified.tflite"
    model_obj.write_model(modified_model_path)

    auc = run_inference(y_trues, all_data, modified_model_path)
    print(f"Attacked model ({args.defend=}) AUC: {auc}")

    os.remove(modified_model_path)
