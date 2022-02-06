import random
import paddle.optimizer as optim
import numpy as np
import paddle
import paddle.nn as nn
from reprod_log import ReprodLogger
from Utils.utils import strLabelConverter
from data.lmdb_dataset import TrainLoader
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

paddle.seed(1)
np.random.seed(1)
random.seed(1)

if __name__ == "__main__":
    # load model
    # def logger
    fp = open('./step8_backforward_paddle.txt', mode='w+')

    from Nets.model import Model as paddle_model
    from config import get_config

    configs = get_config()
    paddle_model = paddle_model(configs)

    # load
    model_state = paddle.load('./PreModel/pre_paddle_model/pren.pdparams')
    paddle_model.set_dict(model_state)
    paddle_model.eval()

    ##define loss
    criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=0)

    # Define optimizer and lr_scheduler
    optimizer = optim.Adam(learning_rate=0.001,
                           parameters=filter(lambda x: not x.stop_gradient, paddle_model.parameters()),
                           weight_decay=configs.TRAIN.weight_decay)

    # encode fake_texts
    with open(configs.DATA.alphabet) as f:
        alphabet = f.readline().strip()
    strconvert = strLabelConverter(alphabet)


    #data
    dataloader_train = TrainLoader(configs)

    for batch_id, (imgs, texts) in enumerate(dataloader_train):
        # read or gen fake data
        fake_imgs = paddle.to_tensor(imgs, dtype=paddle.float32)
        targets = strconvert.encode(texts)
        # forward
        logits = paddle_model(fake_imgs)
        # 计算loss的值
        loss = criterion(paddle.reshape(logits, shape=[-1, logits.shape[-1]]), paddle.reshape(targets, shape=[-1]))
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()
        print(loss.cpu().detach().numpy())
        fp.write(str(loss.cpu().detach().numpy())+"\n")

    fp.close()


