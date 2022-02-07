#   Copyright (c) 2021 PPViT Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Configuration

Configuration for data, model archtecture, and training, etc.
Config can be set by .yaml file or by argparser(limited usage)


"""
import os
from yacs.config import CfgNode as CN
import yaml

_C = CN()
_C.BASE = ['']


_C.DATA = CN()
_C.DATA.alphabet = './data/alphabet_en.txt'
f = open(_C.DATA.alphabet, 'r')
l = f.readline().rstrip()
f.close()
_C.DATA.n_class = len(l) + 3  # pad, unk, eos


# mj and st _Training data settings
_C.LMDB = CN()
_C.LMDB.trainData_dir = "./dataset/lmdb/training"  #训练数据集加载路径
_C.LMDB.imgH = 64
_C.LMDB.imgW = 256
_C.LMDB.trainWorkers = 8
_C.LMDB.aug_prob = 0.3
_C.LMDB.random_seed = 1
_C.LMDB.trainShuffle = True
_C.LMDB.trainBatchsize = 128
_C.LMDB.use_shared_memory = False
_C.LMDB.testDir="./dataset/evaluation/SVT"   #测试数据集加载路径
_C.LMDB.testShuffle=False
_C.LMDB.testBatchsize=1
_C.LMDB.testWorkers=0
# _C.LMDB.testResume="F:\pycharm_project\paddle_pren-main\PreModel\pre_paddle_model\pren" #加载转换torch->paddle转换权重
_C.LMDB.testResume='./output/resume2/Epoch-8-Loss-0.18709387933158853'   #测试数据集 加载权重




# model settings
_C.MODEL = CN()
_C.MODEL.n_class = _C.DATA.n_class
_C.MODEL.max_len = 25
_C.MODEL.n_r = 5  # # number of primitive representations
_C.MODEL.d_model = 384
_C.MODEL.dropout = 0.1

# training settings
_C.TRAIN = CN()
_C.TRAIN.n_epochs = 8
_C.TRAIN.lr = 0.5
_C.TRAIN.lr_milestones = [2, 5, 7]
_C.TRAIN.lr_gammas = [0.2, 0.1, 0.1]
_C.TRAIN.weight_decay = 0.
_C.TRAIN.last_epoch = 0  #断点训练 ，从头训练epoch=0
_C.TRAIN.checkpoints = False  #只加载权重断点训练
_C.TRAIN.resume =False #加载权重和优化器，断点训练
_C.TRAIN.displayInterval = 10000 

# test settings
_C.TEST = CN()
_C.TEST.display = False  #测试时是否一条条显示
_C.TEST.vert_test = True
_C.TEST.batchsize = 1 #batchsize只能为1
_C.TEST.workers = 0

# misc
_C.saveLogdir = "./output/log2"  #日志保存路径
_C.saveFreq = 1  # freq to save chpt ？个epoch保存一次权重
_C.report_Freq = 2000  # freq to logging info
_C.SEED = 1
_C.saveModel = "./output/resume2" #权重保存路径
# _C.VALIDATE_FREQ = 100  # freq to do validation
# _C.EVAL = False  # run evaluation only
# _C.AMP = False  # mix precision training
# _C.LOCAL_RANK = 0
# _C.NGPUS = -1



def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as infile:
        yaml_cfg = yaml.load(infile, Loader=yaml.FullLoader)
    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print('merging config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def get_config(cfg_file=None):
    """Return a clone of config or load from yaml file"""
    config = _C.clone()
    if cfg_file:
        _update_config_from_file(config, cfg_file)
    return config
