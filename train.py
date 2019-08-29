import argparse
from keras import backend as k
import tensorflow as tf

from model.model import Hourglass

from config import config
from utils.heatmap_process import post_process_heatmap

import cv2
import numpy as np
import os

num_classes = config.model.NUM_CLASSES
num_stages = config.model.NUM_STAGES
model = config.model.MODEL_PATH
weights = config.model.MODEL_WEIGHTS_PATH
data_out = config.data.DATA_OUT
data_in = config.data.DATA_IN

if __name__ == '__main__':

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    k.tensorflow_backend.set_session(tf.Session(config=config))

    xnet = Hourglass(num_classes=num_classes, num_stacks=num_stages, num_channels=256, inres=(data_in, data_in), outres=(data_out, data_out))
    xnet.load_trained_model(model, weights)

    xnet.build_model()
    xnet.resume_weights()
    xnet.train()
