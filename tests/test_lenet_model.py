from pathlib import Path
import sys
import unittest
from unittest import TestCase

sys.path.append(str(Path(Path.cwd().parents[0])))
from cnn_arch import lenet_model

class TestLeNet(TestCase):
    def test_build_model(self):
        """Check Error
            When CLASSES has wrong arg, assert TypeError
            When IMAGE_SIZE has wrong arg, assert ValueError
        """
        # test_arg is the pattern of error at arg in build_model
        wrong_arg = [
            ("classes", "b_show")
        ]
        with self.assertRaises(TypeError):
            lenet_model.build_model(CLASSES=wrong_arg[0][0])
        with self.assertRaises(ValueError):
            lenet_model.build_model(IMAGE_SIZE=wrong_arg[0][1])


    def test_build_model_immutable(self):
        """Check Error
            When CLASSES has wrong arg, assert TypeError
            When IMAGE_SIZE has wrong arg, assert ValueError
        """
        # test_arg is the pattern of error at arg in build_model
        wrong_arg = [
            ("classes", "b_show")
        ]
        with self.assertRaises(TypeError):
            lenet_model.build_model_immutable(CLASSES=wrong_arg[0][0])
        with self.assertRaises(ValueError):
            lenet_model.build_model_immutable(IMAGE_SIZE=wrong_arg[0][1])


if __name__=="__main__":
    unittest.main()
