import argparse
from keras import backend as k
import tensorflow as tf

from model import Hourglass

from config import config

num_classes = config.model.NUM_CLASSES
num_stages = config.model.NUM_STAGES
model = config.model.MODEL_PATH
weights = config.model.MODEL_WEIGHTS_PATH

if __name__ == '__main__':

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    k.tensorflow_backend.set_session(tf.Session(config=config))

    xnet = Hourglass(num_classes=num_classes, num_stacks=num_stages, num_channels=256, inres=(256, 256), outres=(64, 64))

    #xnet.build_model(show=True)
    #xnet.train(model_json=model_path, model_weights=weights)

    xnet.load_trained_model(model, weights)
    #xnet.resume_weights()

    out, scale = xnet.inference_file("/home/nikita/1.jpeg")

    import matplotlib.pyplot as plt
    import numpy as np
    import cv2

    first = out.reshape(64, 64, 16)
    for i in range(16):
        img = first[:, :, i]
        img = cv2.resize(img, (640, 480))
        cv2.imshow("test", (img).astype(np.float32))
        cv2.waitKey(0)
