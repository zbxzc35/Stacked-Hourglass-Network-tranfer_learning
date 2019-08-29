from model.create_net import *
from keras.callbacks import CSVLogger, ModelCheckpoint, Callback
from keras.models import load_model, model_from_json
from keras.optimizers import Adam, RMSprop
from keras.losses import mean_squared_error
from keras import backend as k


from utils.data_process import normalize

from matplotlib.pyplot import imread
import scipy

import pandas as pd
from utils.generator import MY_Generator
from config import config


csv_path = config.data.CSV_PATH
batch_size = config.data.BATCH_SIZE
data_set_folder = config.data.FOLDER
data_out = config.data.DATA_OUT
data_in = config.data.DATA_IN
num_stages = config.model.NUM_STAGES


class Hourglass(object):

    def __init__(self, num_classes, num_stacks, num_channels, inres, outres):
        self.num_classes = num_classes
        self.num_stacks = num_stacks
        self.num_channels = num_channels
        self.inres = inres
        self.outres = outres


    def build_model(self, show=False):
        print("*building the model\n")
        self.model = create_hourglass_network(self.num_classes, self.num_stacks,
                                                self.num_channels, self.inres, self.outres)


    def load_trained_model(self, model_json, model_weights):
        print("*loading pretrained model\n")
        with open(model_json) as f:
            self.loaded_model = model_from_json(f.read())
        self.loaded_model.load_weights(model_weights)


    def resume_weights(self):
        print("*coping pretrained weights to new model\n")
        trained_weights = self.loaded_model.get_weights()
        trained_weights = trained_weights[:324] + trained_weights[324:] * (self.num_stacks - 1)
        weights = self.model.get_weights()
        for i in range(len(weights)):
            if weights[i].shape == trained_weights[i].shape:
                weights[i] = trained_weights[i]
        self.model.set_weights(weights)

        for layer_pretrained, layer in zip(self.loaded_model.layers, self.model.layers):
            shape1_1 = layer_pretrained.output_shape
            shape2_1 = layer.output_shape
            shape1_2 = layer_pretrained.input_shape
            shape2_2 = layer.input_shape
            if shape1_1 == shape2_1 and shape1_2 == shape2_2:
                layer.trainable = False
                continue
            print(layer.name + " was changed and set to be trainable")


    def train(self):
        print("*preparing data\n")
        data = pd.read_csv(csv_path)
        labels = data.drop("name", axis=1).values
        imgs = data_set_folder + data["name"].values
        gen = MY_Generator(imgs, labels, batch_size=batch_size, size_in=data_in, size_out=data_out, num_stages=num_stages)

        b = ModelCheckpoint("check_point.h5", monitor='val_loss', save_best_only=True, save_weights_only=True)

        K.set_value(self.model.optimizer.lr, 5e-4)
        self.model.fit_generator(generator=gen, epochs=2)
        for layer in self.model.layers:
            layer.trainable = True

        K.set_value(self.model.optimizer.lr, 5e-5)
        self.model.fit_generator(generator=gen, epochs=50, callbacks = [b], validation_data=gen)
