from easydict import EasyDict as edict

config = edict()

config.data = edict()
config.model = edict()
config.eval = edict()

config.data.BATCH_SIZE = 11
config.data.FOLDER = "data_set/temp/"
config.data.CSV_PATH = "data_set/temp.csv"
config.data.DATA_IN = 256
config.data.DATA_OUT = 64

config.model.NUM_STAGES = 3
config.model.NUM_CLASSES = 3 + 1
config.model.MODEL_PATH = "/home/nikita/hourglass/model/net_arch.json"
config.model.MODEL_WEIGHTS_PATH = "/home/nikita/hourglass/model/weights_epoch96.h5"

config.eval.MODEL_WEIGHTS_PATH = "/home/nikita/hourglass/model/test.h5"
config.eval.NUM_CLASSES = 1 + 1
config.eval.NUM_STAGES = 2
config.eval.DATA_IN = 256
config.eval.DATA_OUT = 64
config.eval.VIDEO_PATH = "/home/nikita/Downloads/229/cam1.avi"
