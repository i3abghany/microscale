import os
import sys

sys.path.insert(0, "..")
import numpy as np
import tensorflow as tf
from model_object import ModelObject
from utils import get_argparser

IMAGE_SIZE = 96
BATCH_SIZE = 64
BATCHES = 10
DATASET_NAME = 'vw_coco2014_96'

def get_data(data_dir):

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=.1,
        horizontal_flip=True,
        validation_split=0.1,
        rescale=1. / 255)

    test_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        subset='validation',
        color_mode='rgb',
        seed=1)

    test_data = []
    for _ in range(BATCHES):
        eval_data = next(test_generator, None)
        test_data.append(eval_data)

    return test_data

def run_inference(model_path, test_data):
    interpreter = tf.lite.Interpreter(model_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    output_data = []
    labels = []
    input_scale, input_zero_point = input_details[0]["quantization"]
    for eval_data in test_data:
        for dat, label in zip(eval_data[0], eval_data[1]):
            dat = dat.reshape((1, 96, 96, 3))
            dat_q = np.array(dat/input_scale + input_zero_point, dtype=np.int8)
            interpreter.set_tensor(input_details[0]['index'], dat_q)
            interpreter.invoke()
            output_data.append(np.argmax(interpreter.get_tensor(output_details[0]['index'])))
            labels.append(np.argmax(label))

    num_correct = np.sum(np.array(labels) == output_data)
    acc = num_correct / len(labels)
    return acc

if __name__ == '__main__':
    args = get_argparser().parse_args()

    model_obj = ModelObject(args.model_path, defended=args.defend)
    batches = get_data(os.path.join(args.data_dir, DATASET_NAME))

    baseline_acc = run_inference(args.model_path, batches)
    print(f"Baseline AUC: {baseline_acc}")

    for _ in range(args.n_bits):
        model_obj.flip_bit(rand=args.random)

    if args.defend:
        model_obj.check_integrity(verbose=args.verbose)

    modified_model_path = "./quant_model_modified.tflite"
    model_obj.write_model(modified_model_path)

    acc = run_inference(modified_model_path, batches)
    print(f"Attacked model ({args.defend=}) : {acc}")

    os.remove(modified_model_path)
