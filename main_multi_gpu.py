import logging
import os
import random
import sys
import time
import paddle.distributed as dist
import numpy as np
import paddle
import paddle.nn as nn
import paddle.optimizer as optim

from Nets.model import Model
from Utils.lr_op import MultiEpochDecay
from Utils.utils import AverageMeter, strLabelConverter
from config import get_config
from data.lmdb_dataset import TrainLoader
import warnings

paddle.seed(1)
np.random.seed(1)
random.seed(1)

warnings.filterwarnings("ignore")

log_format = "%(asctime)s %(message)s"
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt="%m%d %I:%M:%S %p")

# get default config
configs = get_config()

# set output folder
configs.savedir = '{}/train-{}'.format(configs.saveLogdir, time.strftime('%Y%m%d-%H-%M-%S'))

if not os.path.exists(configs.saveLogdir):
    os.makedirs(configs.saveLogdir, exist_ok=True)

# set logging format
logger = logging.getLogger()
fh = logging.FileHandler(os.path.join(configs.saveLogdir, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logger.addHandler(fh)



def train(dataloader,
          model,
          criterion,
          optimizer,
          epoch,
          total_batch,
          report_Freq=100,
          accum_iter=1,
          displayInterval=10,
          strconvert=None):
    model.train()
    train_loss_meter = AverageMeter()
    time_epoch = time.time()
    time_batch = time.time()
    for batch_id, (imgs, texts) in enumerate(dataloader):
        targets = strconvert.encode(texts)  # [B, L]
        logits = model(imgs)  # [b,25,39]
        loss = criterion(paddle.reshape(logits, shape=[-1, logits.shape[-1]]), paddle.reshape(targets, shape=[-1]))
        loss.backward()
        # default 'reduction' param in nn.CrossEntropyLoss is set to 'mean'
        # loss =  loss / accum_iter

        optimizer.step()
        optimizer.clear_grad()

        batch_size = imgs.shape[0]
        train_loss_meter.update(loss.numpy()[0], batch_size)

        if batch_id % displayInterval == 0:
            pred = logits[0].detach().argmax(1)
            pred = list(pred.cpu().numpy())
            if 1 in pred:
                pred = pred[:pred.index(1)]
            pred = strconvert.decode(pred)
            texts[0] = str(texts[0]).strip('')
            print('[{}] [{}/{}] , loss = {}, gt={} ,  pred={}'.format(epoch, batch_id, len(dataloader), loss.numpy(),
                                                                      texts[0], pred))

        if batch_id % report_Freq == 0:
            logger.info(
                f"Epoch[{epoch:03d}/{configs.TRAIN.n_epochs:03d}], " +
                f"Step[{batch_id:04d}/{total_batch:04d}], " +
                f"Avg Loss: {train_loss_meter.avg:.4f}, " +
                f"Left_times:{(time.time() - time_batch) * (total_batch - batch_id) / (3600)} hours")

        time_batch = time.time()
    train_time = time.time() - time_epoch
    return train_loss_meter.avg, train_time


def main_worker(*args):
    dist.init_parallel_env()  # 第一处修改
    last_epoch = configs.TRAIN.last_epoch
    world_size = paddle.distributed.get_world_size()
    local_rank = paddle.distributed.get_rank()
    logger.info(f'----- world_size = {world_size}, local_rank = {local_rank}')
    seed = configs.SEED + local_rank
    paddle.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # 1. Create model  #处改动，增加paddle.DataParallel封装
    model = Model(configs)
    model = paddle.DataParallel(model)

    # 2. Create train and val dataloader
    dataloader_train = TrainLoader(configs)
    # dataloader_val = get_dataloader(config, dataset_val, 'test', True)
    total_batch_train = len(dataloader_train)
    # total_batch_val = len(dataloader_val)
    logging.info(f'----- Total # of train batch (single gpu): {total_batch_train}')
    # logging.info(f'----- Total # of val batch (single gpu): {total_batch_val}')

    # 3. Define criterion
    criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=0)

    # 4. Define optimizer and lr_scheduler
    scheduler = MultiEpochDecay(learning_rate=configs.TRAIN.lr, milestones=configs.TRAIN.lr_milestones,
                                gamma=configs.TRAIN.lr_gammas)
    optimizer = optim.Adadelta(learning_rate=scheduler,
                               parameters=filter(lambda x: not x.stop_gradient, model.parameters()), rho=0.9,
                               weight_decay=configs.TRAIN.weight_decay)

    # 5. Load pretrained model / load resumt model and optimizer states
    if configs.TRAIN.checkpoints:
        if (configs.TRAIN.checkpoints).endswith('.pdparams'):
            raise ValueError(f'{configs.TRAIN.checkpoints} should not contain .pdparams')
        assert os.path.isfile(configs.TRAIN.checkpoints + '.pdparams') is True
        model_state = paddle.load(configs.TRAIN.checkpoints + '.pdparams')
        model.set_dict(model_state)
        logger.info(f"----- Pretrained: Load model state from {configs.TRAIN.checkpoints}")

    if configs.TRAIN.resume:
        assert os.path.isfile(configs.TRAIN.resume + '.pdparams') is True
        assert os.path.isfile(configs.TRAIN.resume + '.pdopt') is True
        model_state = paddle.load(configs.TRAIN.resume + '.pdparams')
        model.set_dict(model_state)
        opt_state = paddle.load(configs.TRAIN.resume + '.pdopt')
        optimizer.set_state_dict(opt_state)
        logger.info(f"----- Resume Training: Load model and optmizer states from {configs.TRAIN.resume}")

    # 6. Start training
    logging.info(f"Start training from epoch {last_epoch + 1}.")
    with open(configs.DATA.alphabet) as f:
        alphabet = f.readline().strip()
    strconvert = strLabelConverter(alphabet)

    for epoch in range(last_epoch + 1, configs.TRAIN.n_epochs + 1):
        # train
        logging.info(f"Now training epoch {epoch}. LR={optimizer.get_lr():.6f}")
        train_loss, train_time = train(dataloader=dataloader_train, model=model, criterion=criterion,
                                       optimizer=optimizer,
                                       epoch=epoch, total_batch=total_batch_train, report_Freq=configs.report_Freq,
                                       accum_iter=configs.TRAIN.accum_iter,
                                       displayInterval=configs.TRAIN.displayInterval, strconvert=strconvert)

        scheduler.step()

        logger.info(
            f"----- Epoch[{epoch:03d}/{configs.TRAIN.n_epochs:03d}], " + f"Train Loss: {train_loss:.4f}, " + f"time: {train_time:.2f}")

        # model save
        if local_rank == 0:
            if epoch % configs.saveFreq == 0 or epoch == configs.TRAIN.n_epochs:
                if not os.path.exists(configs.saveModel):
                    os.makedirs(configs.saveModel)
                model_path = os.path.join(configs.saveModel, f"Epoch-{epoch}-Loss-{train_loss}")
                paddle.save(model.state_dict(), model_path + '.pdparams')
                paddle.save(optimizer.state_dict(), model_path + '.pdopt')
                logger.info(f"----- Save model: {model_path}.pdparams")
                logger.info(f"----- Save optim: {model_path}.pdopt")


def main():
    main_worker()


if __name__ == "__main__":
    main()
