import tensorflow as tf
import random

from utils import get_weight_ranges, model_flip_bit
from check import clip_in_range


class ModelObject:
    def __init__(self, model):
        self.model = model
        self.weights = model.get_weights()
        self.weight_indices = range(len(self.weights))
        self.lengths = [w.size for w in self.weights]
        self.min_weight, self.max_weight = get_weight_ranges(model)

    def get_model(self):
        return self.model

    def flip_bit(self, exp_only=False, msb_only=False, mantissa_only=False, verbose=False):
        layer_idx = random.choices(self.weight_indices, weights=self.lengths, k=1)[0]
        weight_idx = random.randint(0, self.weights[layer_idx].size - 1)
        bit_idx = self._get_bit_idx(exp_only, msb_only, mantissa_only)

        if verbose:
            print(f"Flipping bit {bit_idx} in layer {layer_idx}, weight {weight_idx}")

        model_flip_bit(self.model, layer_idx, weight_idx, bit_idx)


    @staticmethod
    def _get_bit_idx(exp_only, msb_only, mantissa_only):
        if exp_only:
            return random.randint(22, 31)
        elif mantissa_only:
            return random.randint(0, 21)
        elif msb_only:
            return 30
        else:
            return random.randint(0, 31)


    def clip(self):
        clip_in_range(self.model, self.min_weight, self.max_weight)
