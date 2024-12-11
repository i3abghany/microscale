import os
import sys
import numpy as np
import tensorflow as tf

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from utils import get_argparser
from model_object import ModelObject


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
    if not os.path.exists(args.model_path):
        print(f"Model file not found: {args.model_path}")
        sys.exit(-1)

    obj = ModelObject(tf.keras.models.load_model(args.model_path))
    model = obj.get_model()

    acc = model.evaluate(x=test_data, y=test_labels, batch_size=64, verbose=1, return_dict=True)["accuracy"]
    print(f"Accuracy before flipping: ", acc)

    for i in range(args.n_bits):
        obj.flip_bit(args.exp_only, args.msb_only, args.mantissa_only, args.verbose)

        acc = model.evaluate(x=test_data, y=test_labels, batch_size=64, verbose=0, return_dict=True)["accuracy"]
        print(f"Accuracy after {i+1} bits: ", acc)

        if args.defend:
            obj.clip()
            acc = model.evaluate(x=test_data, y=test_labels, batch_size=64, verbose=0, return_dict=True)["accuracy"]
            print(f"Accuracy with defence: ", acc)
