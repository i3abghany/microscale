import os
import sys
import tensorflow as tf
from tensorflow import keras

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from utils import get_argparser
from model_object import ModelObject

IMAGE_SIZE = 96
BATCH_SIZE = 120
DATASET_NAME = "vw_coco2014_96"


def get_data(data_dir):

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=0.1,
        horizontal_flip=True,
        validation_split=0.1,
        rescale=1.0 / 255,
    )

    test_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        subset="validation",
        color_mode="rgb",
        seed=1,
    )

    return test_generator


if __name__ == "__main__":
    args = get_argparser().parse_args()
    if not os.path.exists(args.model_path):
        print(f"Model file not found: {args.model_path}")
        sys.exit(-1)

    test_gen = get_data(os.path.join(args.data_dir, DATASET_NAME))
    obj = ModelObject(keras.models.load_model(args.model_path))

    acc = obj.get_model().evaluate(test_gen, return_dict=True)
    print(f"Performance before flipping: ", acc["accuracy"])
    for i in range(args.n_bits):
        obj.flip_bit(args.exp_only, args.msb_only, args.mantissa_only, args.verbose)
        acc = obj.get_model().evaluate(test_gen, return_dict=True)
        print(f"Performance after {i+1} bits: ", acc["accuracy"])

        if args.defend:
            obj.clip()
            acc = obj.get_model().evaluate(test_gen, return_dict=True)
            print(f"Performance with defence: ", acc["accuracy"])
