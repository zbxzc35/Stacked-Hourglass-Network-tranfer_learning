from create_net import *
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.models import load_model, model_from_json
from keras.optimizers import Adam, RMSprop
from keras.losses import mean_squared_error

from data_process import normalize

from matplotlib.pyplot import imread
import scipy


class Hourglass(object):

    def __init__(self, num_classes, num_stacks, num_channels, inres, outres):
        self.num_classes = num_classes
        self.num_stacks = num_stacks
        self.num_channels = num_channels
        self.inres = inres
        self.outres = outres


    def build_model(self, show=False):
        self.model = create_hourglass_network(self.num_classes, self.num_stacks,
                                                self.num_channels, self.inres, self.outres)
        if show:
            self.model.summary()


    def load_trained_model(self, model_json, model_weights):
        with open(model_json) as f:
            self.loaded_model = model_from_json(f.read())
        self.loaded_model.load_weights(model_weights)

        print(self.loaded_model.summary())

    def resume_weights(self):
        for layer in self.loaded_model.layers:
            print(layer.name)


    def inference_rgb(self, rgbdata, orgshape, mean=None):

        scale = (orgshape[0] * 1.0 / self.inres[0], orgshape[1] * 1.0 / self.inres[1])
        imgdata = scipy.misc.imresize(rgbdata, self.inres)

        if mean is None:
            mean = np.array([0.4404, 0.4440, 0.4327], dtype=np.float)

        imgdata = normalize(imgdata, mean)

        input = imgdata[np.newaxis, :, :, :]

        #out = self.loadmodel.predict(input)
        out = self.loaded_model.predict(input)
        return out[-1], scale

    def inference_file(self, imgfile, mean=None):
        imgdata = imread(imgfile)
        ret = self.inference_rgb(imgdata, imgdata.shape, mean)
        return ret
