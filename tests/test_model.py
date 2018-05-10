import numpy as np
from numpy.testing import assert_allclose
import os
from pathlib import Path
from skimage.io import imread
import sys
import tempfile
import unittest
from unittest import TestCase

from keras.utils import np_utils
from keras.models import save_model, load_model

sys.path.append(str(Path(Path.cwd().parents[0])))
from cnn_arch import lenet_model

class TestModel(TestCase):
    def test_model_arch(self):
        """Check Error
            When CLASSES has wrong arg, assert TypeError
            When IMAGE_SIZE has wrong arg, assert ValueError
        """
        # test_arg is the pattern of error at arg in build_model
        wrong_arg = [
            ("classes", "b_show")
        ]
        with self.assertRaises(TypeError):
            lenet_model.model_arch(CLASSES=wrong_arg[0][0])
        with self.assertRaises(ValueError):
            lenet_model.model_arch(IMAGE_SIZE=wrong_arg[0][1])


    def test_model_arch_immutable(self):
        pass


if __name__=="__main__":
    unittest.main()
