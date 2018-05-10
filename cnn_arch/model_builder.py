"""This code make model"""
from cnn_arch import lenet_model


# build various model
def build_model(MODEL_NAME="lenet",
                b_immutable=False,
                CLASSES=2,
                IMAGE_SIZE=128,
                b_show=False,
                b_gray=False):
    # make model depending on MODEL_NAME
    if MODEL_NAME=="lenet":
        model = lenet_model.model_arch(CLASSES=CLASSES,
                            IMAGE_SIZE=IMAGE_SIZE,
                            b_show=b_show,
                            b_gray=b_gray)
    else:
        # if this code has no model depending on MODEL_NAME, execute lenet_model
        model = lenet_model.model_arch(CLASSES=CLASSES,
                            IMAGE_SIZE=IMAGE_SIZE,
                            b_show=b_show,
                            b_gray=b_gray)
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


def batch_model(MODEL_NAME="lenet",
                b_immutable=False,
                CLASSES=2,
                IMAGE_SIZE=128,
                b_show=False,
                b_gray=False,
                LOSS='categorical_crossentropy',
                OPTIMIZER='adam',
                METRICS=['acc']):
    model = build_model(MODEL_NAME=MODEL_NAME,
                        b_immutable=b_immutable,
                        CLASSES=CLASSES,
                        IMAGE_SIZE=IMAGE_SIZE,
                        b_show=b_show,
                        b_gray=b_gray)
    model = model_compile(model,
                          LOSS=LOSS,
                          OPTIMIZER=OPTIMIZER,
                          METRICS=METRICS)
    return model
