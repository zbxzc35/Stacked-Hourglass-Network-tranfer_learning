from easydict import EasyDict as edict

config = edict()

config.data = edict()
config.model = edict()

config.data.BATCH_SIZE = 8

config.model.NUM_STAGES = 2
config.model.NUM_CLASSES = 8
config.model.MODEL_PATH = "/home/nikita/hourglass/model/net_arch.json"
config.model.MODEL_WEIGHTS_PATH = "/home/nikita/hourglass/model/weights_epoch96.h5"

#config.model.MODEL_PATH = "/home/nikita/hourglass/model/net_arch_mobile.json"
#config.model.MODEL_WEIGHTS_PATH = "/home/nikita/hourglass/model/mobile_weights_epoch99.h5"
