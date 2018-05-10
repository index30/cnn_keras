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
        pass


    """
    test whether training model's result and saving model's result is match
    """
    def test_model_saving(self):
        # build and compile models
        # build from lenet_model.py, build_model_immutable
        model = lenet_model.build_model_immutable()
        # compile model
        model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['acc']
        )

        # Get path of image for training
        data_path = Path(Path.cwd().parent, 'data/tests').glob('*.jpg')
        """data_path
        dog.jpg   ...label 0
        cat1~3.jpg...label 1
        """
        X = [imread(str(path)) for path in data_path]
        X = np.array(X)
        y = np.array([1, 1, 1, 0])
        # convert label to one-hot label
        y = np_utils.to_categorical(y)
        # training
        model.train_on_batch(X, y)

        out = model.predict(X[0][np.newaxis, :])
        _, fname = tempfile.mkstemp('.h5')
        save_model(model, fname)

        new_model = load_model(fname)
        os.remove(fname)

        out2 = new_model.predict(X[0][np.newaxis, :])
        print(out)
        print(out2)
        assert_allclose(out, out2, atol=1e-05)


if __name__=="__main__":
    unittest.main()
