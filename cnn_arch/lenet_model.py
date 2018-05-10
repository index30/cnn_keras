"""This code define the model of LeNet"""

from datetime import datetime

from keras.models import Sequential
from keras.layers import (Conv2D, MaxPooling2D,
                          Activation, Dropout, Flatten, Dense, Input)

"""build model
input: b_show  ... boolean whether display the architecture of LeNet
       CLASSES ... The number which you want to classify img

about model:
    Conv -> Pool -> Conv -> Pool -> Dense -> Dense
    using activation function is relu
    using dropout
    using softmax when generate result

output: model ... The architecture of LeNet
"""
def build_model(CLASSES=2, IMAGE_SIZE=128, b_show=False, b_gray=False):
    # Definition of model
    model = Sequential()
    if b_gray:
        input_shape = (IMAGE_SIZE, IMAGE_SIZE, 1)
    else:
        input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)
    # 第一層
    model.add(Conv2D(32, (5, 5), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # 第二層
    model.add(Conv2D(64, (5, 5)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # 全結合層
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(CLASSES))
    model.add(Activation('softmax'))
    if b_show:
        model.summary()
    return model

def model_compile(model,
                  LOSS='categorical_crossentropy',
                  OPTIMIZER='adam',
                  METRICS=['acc']):
    model.compile(
        loss=LOSS,
        optimizer=OPTIMIZER,
        metrics=METRICS
    )
    return model

def batch_build(CLASSES=2,
                IMAGE_SIZE=128,
                b_show=False,
                b_gray=False,
                LOSS='categorical_crossentropy',
                OPTIMIZER='adam',
                METRICS=['acc']):
    model = build_model(CLASSES=CLASSES,
                        IMAGE_SIZE=IMAGE_SIZE,
                        b_show=b_show,
                        b_gray=b_gray)
    model = model_compile(model,
                          LOSS=LOSS,
                          OPTIMIZER=OPTIMIZER,
                          METRICS=METRICS)
    return model


"""build_model_immutable
This method is similar to build_model,
but this method return certain model
"""
def build_model_immutable():
    # Definition of model
    model = Sequential()
    input_shape = (128, 128, 3)
    # 第一層
    model.add(Conv2D(32, (5, 5), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # 第二層
    model.add(Conv2D(64, (5, 5)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # 全結合層
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2))
    model.add(Activation('softmax'))

    return model
