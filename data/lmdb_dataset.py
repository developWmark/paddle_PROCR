# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import random

import cv2
import lmdb
import numpy as np
import paddle
import paddle.vision.transforms as transforms
from paddle.io import Dataset, DataLoader

from config import get_config
from data.data_utils import Augmenter
from data.data_utils import ToPILImage

paddle.seed(1)
np.random.seed(1)
random.seed(1)

#train dataset
class LMDBDataSet(Dataset):
    def __init__(self, configs):
        super(LMDBDataSet, self).__init__()
        data_dir = configs.LMDB.trainData_dir
        self.do_shuffle = configs.LMDB.trainShuffle
        self.lmdb_sets = self.load_hierarchical_lmdb_dataset(data_dir)
        self.data_idx_order_list = self.dataset_traversal()
        self.aug = Augmenter(p=configs.LMDB.aug_prob)
        if self.do_shuffle:
            np.random.shuffle(self.data_idx_order_list)
        self.transforms = get_train_transforms(configs)


    def load_hierarchical_lmdb_dataset(self, data_dir):
        lmdb_sets = {}
        dataset_idx = 0
        for dirpath, dirnames, filenames in os.walk(data_dir + '/'):
            if not dirnames:
                env = lmdb.open(
                    dirpath,
                    max_readers=32,
                    readonly=True,
                    lock=False,
                    readahead=False,
                    meminit=False)
                txn = env.begin(write=False)
                num_samples = int(txn.get('num-samples'.encode()))
                lmdb_sets[dataset_idx] = {"dirpath": dirpath, "env": env, "txn": txn, "num_samples": num_samples}
                dataset_idx += 1
        return lmdb_sets

    def dataset_traversal(self):
        lmdb_num = len(self.lmdb_sets)
        total_sample_num = 0
        for lno in range(lmdb_num):
            total_sample_num += self.lmdb_sets[lno]['num_samples']
        data_idx_order_list = np.zeros((total_sample_num, 2))
        beg_idx = 0
        for lno in range(lmdb_num):
            tmp_sample_num = self.lmdb_sets[lno]['num_samples']
            end_idx = beg_idx + tmp_sample_num
            data_idx_order_list[beg_idx:end_idx, 0] = lno
            data_idx_order_list[beg_idx:end_idx, 1] = list(range(tmp_sample_num))
            data_idx_order_list[beg_idx:end_idx, 1] += 1
            beg_idx = beg_idx + tmp_sample_num
        return data_idx_order_list

    def get_img_data(self, value):
        """get_img_data"""
        if not value:
            return None
        imgdata = np.frombuffer(value, dtype='uint8')
        if imgdata is None:
            return None
        imgori = cv2.imdecode(imgdata, 1)
        if imgori is None:
            return None
        return imgori

    def get_lmdb_sample_info(self, txn, index):
        label_key = 'label-%09d'.encode() % index
        label = txn.get(label_key)
        if label is None:
            return None
        label = label.decode('utf-8')
        img_key = 'image-%09d'.encode() % index
        imgbuf = txn.get(img_key)
        return imgbuf, label, img_key, label_key

    def __getitem__(self, idx):
        lmdb_idx, file_idx = self.data_idx_order_list[idx]
        lmdb_idx = int(lmdb_idx)
        file_idx = int(file_idx)
        imgbuf, label, img_key, label_key = self.get_lmdb_sample_info(self.lmdb_sets[lmdb_idx]['txn'],  # imgbuf, label
                                                                      file_idx)

        imageBuf = np.frombuffer(imgbuf, dtype=np.uint8)
        img = cv2.imdecode(imageBuf, cv2.IMREAD_COLOR)

        # 1：error images
        if img is None:
            # print('errot images: {}, use next one.'.format(img_key))
            return self.__getitem__(idx + 1)

        # 2：ignore too small images
        h, w, _ = img.shape
        if min(h, w) <= 5:
            # print('Too small image {}, use next one.'.format(img_key))
            return self.__getitem__(idx + 1)

        # 3：Too long text
        label = str(label).lower()  # 大写字母转小写

        if len(label) >= 25:
            # print('Too long text: {}, use next one.'.format(label_key))
            return self.__getitem__(idx + 1)

        # 4: data preprocess
        img = self.aug.apply(img, len(label))  # h,w,c
        img = self.transforms(img)

        return (img, label)

    def __len__(self):
        return self.data_idx_order_list.shape[0]


def get_train_transforms(configs):
    transforms_train = transforms.Compose([
        ToPILImage(),
        #transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),  # HWC
        transforms.Resize((configs.LMDB.imgH, configs.LMDB.imgW)),  # HWC
        transforms.Transpose(order=(2, 0, 1)),  # CHW
        transforms.Normalize(mean=[0, 0, 0], std=[255.0, 255.0, 255.0]),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    return transforms_train



def TrainLoader(configs):
    lmdb_dataset = LMDBDataSet(configs)
    dataloader = DataLoader(dataset=lmdb_dataset, batch_size=configs.LMDB.trainBatchsize,
                                shuffle=configs.LMDB.trainShuffle,
                                num_workers=configs.LMDB.trainWorkers, use_shared_memory=configs.LMDB.use_shared_memory)
    return dataloader


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    configs = get_config()
    train_loader = TrainLoader(configs)
    for index, data in enumerate(train_loader):
        im, label = data
        print(type(im), type(label))
        plt.imshow(im[0].transpose([1, 2, 0]) * 0.5 + 0.5)
        plt.show()
