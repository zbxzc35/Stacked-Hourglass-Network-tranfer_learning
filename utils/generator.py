import cv2
import numpy as np
import math
from keras.utils import Sequence
from transform import*
from numpy.random import RandomState


class MY_Generator(Sequence):

    def __init__(self, img_paths, labels, batch_size, size_in, size_out, num_stages, flag=False, aug=True):
        self.num_stages = num_stages
        self.size_in = size_in
        self.flag = flag
        self.img_paths = img_paths
        self.labels = labels
        self.batch_size = batch_size
        self.test = 0
        self.size_out = size_out
        self.aug = aug

        PRNG = RandomState()
        self.transform = Compose([
            [ColorJitter(prob=0.5), None],
            Expand((0.8, 1.5)),
            RandomCompose([
                RandomRotate(360),
                RandomShift(0.2)]),
            Scale(512),
            ElasticTransform(300),
            RandomCrop(512),
            ],
            PRNG,
            border='constant',
            fillval=0,
            outside_points='inf')

    def __len__(self):
        if not self.test:
            for i in range(len(self.img_paths) - 1, -1, -1):
                if cv2.imread(self.img_paths[i]) is None:
                    self.img_paths = np.delete(self.img_paths, i, 0)
                    self.labels = np.delete(self.labels, i, 0)
            self.test += 1
        return int(np.ceil(len(self.img_paths) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.img_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        new_batch = []
        for i, batch in enumerate(batch_y):
            new_batch.append([(batch[i], batch[i+1]) for i in range(0, len(batch), 2)])
        batch_y = np.array(new_batch)

        imgs = np.array([cv2.imread(file_name) for file_name in batch_x])

        if np.random.choice([0, 0, 1]) or not self.aug:
            maps = np.array([self.get_heatmap(batch_y[i], imgs[i].shape, self.size_out) for i in range(len(batch_y))])
            imgs = np.array([cv2.resize(im, (self.size_in, self.size_in)) / 255 for im in imgs])
            return imgs, [maps] * self.num_stages
        else:
            data = np.array([self.transform(im, pt) for im, pt in zip(imgs, batch_y)])
            maps = np.array([self.get_heatmap(data[i, 1], data[i, 0].shape, self.size_out) for i in range(len(data[:, 1]))])
            imgs = np.array([cv2.resize(im, (self.size_in, self.size_in)) / 255 for im in data[:, 0]])
            return imgs, [maps] * self.num_stages



    def get_heatmap(self, annos, img_size, size_out):
        num_joints = len(annos) + 1

        joints_heatmap = np.zeros((num_joints, img_size[0], img_size[1]), dtype=np.float32)

        for i, points in enumerate(annos):
            if points[0] < 0 or points[1] < 0 or points[0] >= img_size[1] or points[1] >= img_size[0]:
                continue
            joints_heatmap = self.put_heatmap(joints_heatmap, i, points)

        joints_heatmap[-1, :, :] = np.clip(1 - np.amax(joints_heatmap, axis=0), 0.0, 1.0)
        mapholder = []
        for i in range(num_joints):
            a = cv2.resize(np.array(joints_heatmap[i, :, :]), (size_out, size_out))
            mapholder.append(a)

        mapholder = np.array(mapholder, np.float32).transpose(1, 2, 0)
        return mapholder


    def put_heatmap(self, heatmap, plane_idx, center, sigma=20):
        center_x, center_y = center
        _, height, width = heatmap.shape

        th = 4.6052
        delta = math.sqrt(th * 2)

        x0 = int(max(0, center_x - delta * sigma + 0.5))
        y0 = int(max(0, center_y - delta * sigma + 0.5))

        x1 = int(min(width - 1, center_x + delta * sigma + 0.5))
        y1 = int(min(height - 1, center_y + delta * sigma + 0.5))

        exp_factor = 1 / 2.0 / sigma / sigma

        arr_heatmap = heatmap[plane_idx, y0:y1 + 1, x0:x1 + 1]
        y_vec = (np.arange(y0, y1 + 1) - center_y)**2
        x_vec = (np.arange(x0, x1 + 1) - center_x)**2
        xv, yv = np.meshgrid(x_vec, y_vec)
        arr_sum = exp_factor * (xv + yv)
        arr_exp = np.exp(-arr_sum)
        arr_exp[arr_sum > th] = 0
        heatmap[plane_idx, y0:y1 + 1, x0:x1 + 1] = np.maximum(arr_heatmap, arr_exp)
        return heatmap
