import sys
from tensorflow import keras

"""
This script converts a saved model to a Keras model and saves it to a new file in the same directory.

Usage:
python fp_saved_model_to_keras.py <saved_model_path>

"""
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fp_saved_model_to_keras.py <saved_model_path>")
        exit(1)

    model = keras.models.load_model(sys.argv[1])
    model.summary()
    model_path = sys.argv[1] + ".keras"
    model.save(model_path, save_format="tf")
    print(f"Saved model to {model_path}")
