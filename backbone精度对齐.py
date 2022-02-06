import torch
import paddle
import os
import numpy as np
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 注意事项：
# 1.FC层的权重需要转置；
# 2.如果此处torch_key, paddle_key的名字没有对应上，需要写程序一一对应；
# 3.lstm中paddle存在rnn.1.rnn.0.cell_bw.weight_hh，但是torch无
from paddle import device


def print_model_named_params(model):
    print('---------model.named_parameters()-----------')
    for name, param in model.named_parameters():
        print(name, param.shape)
    print('-------------end---------------------')


def print_model_named_buffers(model):
    print('---------model.named_buffers()-------------')
    for name, param in model.named_buffers():
        print(name, param.shape)
    print('--------------end-----------------')


def torch_to_paddle_mapping():  # 只做了部分映射
    mapping = [
        (f'cnn._conv_stem', f'cnn._conv_stem'),
        (f'cnn._bn0', f'cnn._bn0'),
    ]

    cnn_layer = 26
    for idx in range(0, 26):
        tor_con1 = f'cnn._blocks.{idx}._depthwise_conv'
        tor_con2 = f'cnn._blocks.{idx}._bn1'
        tor_con3 = f'cnn._blocks.{idx}._se_reduce'
        tor_con4 = f'cnn._blocks.{idx}._se_expand'
        tor_con5 = f'cnn._blocks.{idx}._project_conv'
        tor_con6 = f'cnn._blocks.{idx}._bn2'
        tor_con7 = f'cnn._blocks.{idx}._expand_conv'
        tor_con8 = f'cnn._blocks.{idx}._bn0'

        pp_con1 = f'cnn._blocks.{idx}.._depthwise_conv'
        pp_con2 = f'cnn._blocks.{idx}.._bn1'
        pp_con3 = f'cnn._blocks.{idx}.._se_reduce'
        pp_con4 = f'cnn._blocks.{idx}.._se_expand'
        pp_con5 = f'cnn._blocks.{idx}.._project_conv'
        pp_con6 = f'cnn._blocks.{idx}.._bn2'
        pp_con7 = f'cnn._blocks.{idx}.._expand_conv'
        pp_con8 = f'cnn._blocks.{idx}.._bn0'

        layer_mapping = [
            (tor_con1, pp_con1),
            (tor_con2, pp_con2),
            (tor_con3, pp_con3),
            (tor_con4, pp_con4),
            (tor_con5, pp_con5),
            (tor_con6, pp_con6),
            (tor_con7, pp_con7),
            (tor_con8, pp_con8)
        ]
        mapping.extend(layer_mapping)

    aggP_layer = 3
    for pidx in range(1, 4):
        for idx1 in range(0, 5):
            for idx2 in range(0, 5):
                mapping.append((f'agg_p{pidx}.aggs.{idx1}.{idx2}', f'agg_p{pidx}.aggs.{idx1}.{idx2}'))

    agg_w = 3
    for widx in range(1, 4):
        for idx1 in ['n', 'd']:
            for idx2 in range(0, 5):
                mapping.append((f'agg_w{widx}.conv_{idx1}.{idx2}', f'agg_w{widx}.conv_{idx1}.{idx2}'))

    gcn = 1
    gcn_mapping = [
        (f'gcn_pool.conv_n', f'gcn_pool.conv_n'),
        (f'gcn_pool.linear', f'gcn_pool.linear'),
        (f'gcn_weight.conv_n', f'gcn_weight.conv_n'),
        (f'gcn_weight.linear', f'gcn_weight.linear'),
        (f'linear', f'linear')
    ]
    mapping.extend(gcn_mapping)

    return mapping


def convert(torch_model, paddle_model):
    def _set_value(th_name, pd_name):
        th_shape = th_params[th_name].shape
        pd_shape = tuple(pd_params[pd_name].shape)  # paddle shape default type is list
        # assert th_shape == pd_shape, f'{th_shape} != {pd_shape}'
        print(f'**SET** {th_name} {th_shape} **TO** {pd_name} {pd_shape}')
        # if isinstance(th_params[th_name], torch.nn.parameter.Parameter):
        #     value = th_params[th_name].data.numpy()
        # else:
        #     value = th_params[th_name].numpy()
        value = th_params[th_name].cpu().detach().numpy()
        if ("fc" in th_name and "weight" in th_name) or (len(value.shape) == 2 and th_shape[0] == pd_shape[1]):
            value = value.transpose((1, 0))
        new_params[pd_name] = value

    # 1. get paddle and torch model parameters
    pd_params = {}
    th_params = {}
    new_params = {}
    for name, param in paddle_model.named_parameters():
        pd_params[name] = param
    for name, param in torch_model.named_parameters():
        th_params[name] = param

    for name, param in paddle_model.named_buffers():
        pd_params[name] = param
    for name, param in torch_model.named_buffers():
        th_params[name] = param

    # 2. get name mapping pairs
    mapping = torch_to_paddle_mapping()


    # 3. set torch param values to paddle params: may needs transpose on weights
    for th_name, pd_name in mapping:
        if th_name in th_params.keys():  # nn.Parameters 看mapping的映射 全名在话直接赋值
            _set_value(th_name, pd_name)
        else:  # weight & bias &bn的running_mean，running_var
            if f'{th_name}.weight' in th_params.keys():
                th_name_w = f'{th_name}.weight'
                pd_name_w = f'{pd_name}.weight'
                _set_value(th_name_w, pd_name_w)
            if f'{th_name}.bias' in th_params.keys():
                th_name_b = f'{th_name}.bias'
                pd_name_b = f'{pd_name}.bias'
                _set_value(th_name_b, pd_name_b)
            if f'{th_name}.running_var' in th_params.keys():
                th_name_b = f'{th_name}.running_var'
                pd_name_b = f'{pd_name}._variance'
                _set_value(th_name_b, pd_name_b)
            if f'{th_name}.running_mean' in th_params.keys():
                th_name_b = f'{th_name}.running_mean'
                pd_name_b = f'{pd_name}._mean'
                _set_value(th_name_b, pd_name_b)

    paddle_model.set_state_dict(new_params)
    return paddle_model


# 保存paddle权重

def save_model(net, model_path, model_name="", prefix='pren'):
    """
    save model to the target path
    """
    model_path = os.path.join(model_path, model_name)
    model_path = os.path.join(model_path, prefix)

    paddle.save(net.state_dict(), model_path + ".pdparams")
    print("Already save model in {}".format(model_path))


def main():
    # paddle_model
    from Nets.model import Model as paddle_model
    from config import get_config
    Configs = get_config()
    paddle_model = paddle_model(Configs)
    paddle_model.eval()
    print_model_named_params(paddle_model)
    print_model_named_buffers(paddle_model)

    print("&" * 100)

    # torch_model
    device = torch.device('cpu')
    from torch_Nets.model import Model as torch_model
    checkpoint = torch.load("./PreModel/pren.pth", torch.device('cpu'))
    torch_model = torch_model(checkpoint['model_config'])
    torch_model.load_state_dict(checkpoint['state_dict'])
    torch_model = torch_model.to(device)
    torch_model.eval()
    print_model_named_params(torch_model)
    print_model_named_buffers(torch_model)

    # convert weights
    print("=====================convert weights===============================")
    paddle_model = convert(torch_model, paddle_model)


    # check correctness
    x = np.full(shape=(10, 3, 64, 256), fill_value=0.1,dtype='float32')
    x_paddle = paddle.to_tensor(x, dtype='float32')
    x_torch = torch.Tensor(x).to(device)

    torch_model.eval()
    paddle_model.eval()
    out_torch = torch_model(x_torch)
    out_paddle = paddle_model(x_paddle)

    out_torch = out_torch.cpu().detach().numpy()
    out_paddle = out_paddle.cpu().detach().numpy()

    print('torch shape:{}    paddle shape:{}'.format(out_torch.shape, out_paddle.shape))
    print(out_torch[0, 0:100,2])
    print('========================================================')
    print(out_paddle[0, 0:100,2])

    # 保存模型权重
    save_model(paddle_model, model_path="./PreModel", model_name="pre_paddle_model")


    #测试加载
    #####################################################################################
    # from test import test
    # from data.lmdbTestdataset import TestLoader
    # from config import get_config
    # from Utils.utils import strLabelConverter
    #
    # configs = get_config()
    #
    # testloader = TestLoader(configs)
    # print('[Info] Load data from {}'.format(configs.LMDB.testDir))
    #
    # with open(configs.DATA.alphabet) as f:
    #     alphabet = f.readline().strip()
    # strconvert = strLabelConverter(alphabet)
    #
    # test(paddle_model, testloader, converter=strconvert)


if __name__ == '__main__':
    main()
