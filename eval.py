from model.model import Hourglass
from config import config
import cv2
from keras.models import load_model
import numpy as np
from keras import backend as k


weights = config.eval.MODEL_WEIGHTS_PATH
num_classes = config.eval.NUM_CLASSES
num_stages = config.eval.NUM_STAGES
data_out = config.eval.DATA_OUT
data_in = config.eval.DATA_IN
video_path = config.eval.VIDEO_PATH

xnet = Hourglass(num_classes=num_classes, num_stacks=num_stages, num_channels=256, inres=(data_in, data_in), outres=(data_out, data_out))
#xnet.build_model()
#xnet.model.load_weights(weights)
xnet.model = load_model("my_model.h5", compile=False)

out = xnet.model.get_layer("3_conv_1x1_parts").output
inp = xnet.model.layers[0].input
functor = k.function([inp, k.learning_phase()], [out])

cap = cv2.VideoCapture(video_path)

ret, frame = cap.read()
while ret:
    temp = cv2.resize(frame, (data_in, data_in)).reshape(1, data_in, data_in, 3)

    b = functor([temp / 255, 1])[0]
    res1 = np.where(b[0, :, :, 0] == np.amax(b[0, :, :, 0]))
    img = cv2.circle(cv2.resize(frame, (data_in, data_in)), (res1[1][0] * 4, res1[0][0] * 4), 3, (255, 0, 0), 3)

    cv2.imshow("test", img)
    cv2.waitKey(1)

    ret, frame = cap.read()
