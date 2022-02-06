import random

import numpy as np
import paddle
import paddle.nn as nn
from reprod_log import ReprodLogger

from Utils.utils import strLabelConverter

paddle.seed(1)
np.random.seed(1)
random.seed(1)

if __name__ == "__main__":
    # load model
    # def logger
    reprod_logger = ReprodLogger()

    from Nets.model import Model as paddle_model
    from config import get_config

    configs = get_config()
    paddle_model = paddle_model(configs)
    # load
    model_state = paddle.load('./PreModel/pre_paddle_model/pren.pdparams')
    paddle_model.set_dict(model_state)
    paddle_model.eval()

    # define loss
    criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=0)

    # read or gen fake data
    fake_imgs = np.random.random((10, 3, 64, 256))
    fake_texts = ['abcd'] * 10
    fake_imgs = paddle.to_tensor(fake_imgs, dtype=paddle.float32)
    # forward
    logits = paddle_model(fake_imgs)

    with open(configs.DATA.alphabet) as f:
        alphabet = f.readline().strip()
    strconvert = strLabelConverter(alphabet)

    # 计算loss的值
    targets = strconvert.encode(fake_texts)
    loss = criterion(paddle.reshape(logits, shape=[-1, logits.shape[-1]]), paddle.reshape(targets, shape=[-1]))

    #
    reprod_logger.add("loss", loss.cpu().detach().numpy())
    reprod_logger.save("step4_loss_paddle.npy")
