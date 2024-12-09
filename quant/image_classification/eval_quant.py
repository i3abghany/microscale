import os
import sys

sys.path.insert(0, "..")
import numpy as np
import tensorflow as tf
from model_object import ModelObject
from utils import get_argparser


def run_inference(model_path, test_imgs, test_labels):
    interpreter = tf.lite.Interpreter(model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    output_data = []

    predictions = []
    for img in test_imgs:
        input_data = img.reshape(1, 32, 32, 3)
        interpreter.set_tensor(input_details[0]["index"], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]["index"])
        predictions.append(
            output_data.reshape(
                10,
            )
        )
    predictions = np.array(predictions)

    return np.mean(np.argmax(predictions, axis=1) == np.argmax(test_labels, axis=1))


def load_cifar10():
    (_, _), (test_imgs, test_labels) = tf.keras.datasets.cifar10.load_data()

    test_labels = tf.keras.utils.to_categorical(test_labels, 10)

    test_imgs = test_imgs.astype(np.int64)
    test_imgs[test_imgs < 0] += 256
    test_imgs = test_imgs - 128
    test_imgs = test_imgs.astype(np.int8)

    # Randomly sample 1000 images
    np.random.seed(112233)
    sample_indices = np.random.choice(len(test_imgs), 1000)
    test_imgs = test_imgs[sample_indices]
    test_labels = test_labels[sample_indices]

    return test_imgs, test_labels


if __name__ == "__main__":
    args = get_argparser().parse_args()
    model_obj = ModelObject(args.model_path, defended=args.defend)

    test_imgs, test_labels = load_cifar10()
    baseline_acc = run_inference(args.model_path, test_imgs, test_labels)
    print(f"Baseline accuracy: {baseline_acc}")

    for _ in range(args.n_bits):
        model_obj.flip_bit(rand=args.random)

    if args.defend:
        model_obj.check_integrity(verbose=args.verbose)

    modified_model_path = "./quant_model_modified.tflite"
    model_obj.write_model(modified_model_path)

    acc = run_inference(modified_model_path, test_imgs, test_labels)
    print(f"Attacked model ({args.defend=}) accuracy: {acc}")

    os.remove(modified_model_path)
