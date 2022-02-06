import numpy as np
import paddle
from reprod_log import ReprodLogger
import random

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
    model_state = paddle.load('./PreModel/pre_paddle_model/pren.pdparams')
    paddle_model.set_dict(model_state)
    paddle_model.eval()

    # read or gen fake data
    fake_data = np.random.random((10, 3, 64, 256))
    fake_data = paddle.to_tensor(fake_data,dtype=paddle.float32)
    # forward
    out = paddle_model(fake_data)
    #
    reprod_logger.add("logits", out.cpu().detach().numpy())
    reprod_logger.save("step1_forward_paddle.npy")
