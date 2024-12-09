from tensorflow.lite.tools.flatbuffer_utils import (
    read_model_with_mutable_tensors,
    write_model,
)
from utils import get_high_precision_buffers, get_non_null_buffers, get_replicas

import random


class ModelObject:
    """
    Model object class that encapsulates the model object and its redundancy copies.
    Models are assumed in the form of TFLite flatbuffer models.
    """

    def __init__(self, filepath, defended=True):
        self.filepath = filepath
        self.defended = defended
        self.reset_model()

    def flip_bit(self, rand=True):
        """
        Flip a bit in the model object. The bit flip can be positioned in any of the
        buffers in the model object, including the redundancy copies.  The attacker
        is assumed to know the model object structure, including the redundancy
        copies, and whether the redundancy copies are checked for integrity. Hence,
        the attacker is assumed not to flip bits in the redundancy copies if the
        defence is disabled.

        Args:
            random: If True, the bit flip is randomly positioned in any of the
            buffers in the model object (including redundancy copies). If False, the
            bit flip is only in the high-precision buffers.

            defended: If True, redundancy copies are checked against each other to
            ensure that they are consistent. If False, integrity are not checked.
        """
        if rand:
            buf_idx = random.choices(self.non_null_bufs, weights=self.non_null_lengths)[0]
            c = self.model_obj.buffers
        else:
            buf_idx = random.choices(self.hi_precision_bufs, weights=self.hi_precision_lengths)[0]
            if self.defended:
                which_copy = random.randint(0, 2)
                c = [self.model_obj.buffers, self.r1, self.r2][which_copy]
            else:
                c = self.model_obj.buffers

            byte_idx = random.randint(0, len(self.model_obj.buffers[buf_idx].data) - 1)
            bit_idx = random.randint(0, 7)
            c[buf_idx].data[byte_idx] ^= 1 << bit_idx

    def check_integrity(self, verbose=False):
        """
        Check the integrity of the model with redundancy copies. Currently, the
        integrity check is done by majority voting. If the redundancy copies are
        inconsistent, the majority value is used to correct the inconsistency.

        Args:
            verbose: If True, print the inconsistency in the redundancy copies.
        """
        for buf in self.hi_precision_bufs:
            original = self.model_obj.buffers[buf].data
            for i in range(len(original)):
                d = original[i]
                d1 = self.r1[buf].data[i]
                d2 = self.r2[buf].data[i]
                if d != d1 or d1 != d2 or d != d2:
                    if verbose:
                        print("Integrity violation: ", d, d1, d2)
                    majority = (d & d1) | (d1 & d2) | (d2 & d)
                    self.model_obj.buffers[buf].data[i] = majority
                    self.r1[buf].data[i] = majority
                    self.r2[buf].data[i] = majority

    def reset_model(self):
        self.model_obj = read_model_with_mutable_tensors(self.filepath)
        self.hi_precision_bufs, self.hi_precision_lengths = get_high_precision_buffers(self.model_obj)
        self.r1, self.r2 = get_replicas(self.model_obj)
        self.non_null_bufs, self.non_null_lengths = get_non_null_buffers(self.model_obj)

    def write_model(self, filepath):
        write_model(self.model_obj, filepath)
