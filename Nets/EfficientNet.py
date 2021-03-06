import paddle
from paddle import nn
from paddle.nn import functional as F

from Nets.EfficientNet_utils import (
    round_filters,
    round_repeats,
    drop_connect,
    get_same_padding_conv2d,
    get_model_params,
)


class MBConvBlock(nn.Layer):
    """
    Mobile Inverted Residual Bottleneck Block
    Args:
        block_args (namedtuple): BlockArgs, see above
        global_params (namedtuple): GlobalParam, see above
    Attributes:
        has_se (bool): Whether the block contains a Squeeze and Excitation layer.
    """

    def __init__(self, block_args, global_params):
        super().__init__()
        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip  # skip connection and drop connect

        # Get static or dynamic convolution depending on image size
        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)

        # Expansion phase
        inp = self._block_args.input_filters  # number of input channels
        oup = self._block_args.input_filters * self._block_args.expand_ratio  # number of output channels
        if self._block_args.expand_ratio != 1:  #非第一个stage
            w_attr_1, b_attr_1 = self._init_weights()
            self._expand_conv = Conv2d(in_channels=inp, out_channels=oup, kernel_size=1,bias_attr=False,weight_attr=w_attr_1)
            self._bn0 = nn.BatchNorm2D(num_features=oup, momentum=self._bn_mom, epsilon=self._bn_eps)

        # Depthwise convolution phase
        k = self._block_args.kernel_size
        s = self._block_args.stride
        # groups makes it depthwise
        w_attr_2, b_attr_2 = self._init_weights()
        self._depthwise_conv = Conv2d(in_channels=oup, out_channels=oup, groups=oup, kernel_size=k, stride=s, bias_attr=False,weight_attr=w_attr_2)
        self._bn1 = nn.BatchNorm2D(num_features=oup, momentum=self._bn_mom, epsilon=self._bn_eps)

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            w_attr_3, b_attr_3 = self._init_weights()
            w_attr_4, b_attr_4 = self._init_weights()
            num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
            self._se_reduce = Conv2d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1,weight_attr=w_attr_3,bias_attr=b_attr_3)
            self._se_expand = Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1,weight_attr=w_attr_4,bias_attr=b_attr_4)

        # Output phase
        w_attr_4, b_attr_4 = self._init_weights()
        final_oup = self._block_args.output_filters
        self._project_conv = Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias_attr=False,weight_attr=w_attr_4)
        self._bn2 = nn.BatchNorm2D(num_features=final_oup, momentum=self._bn_mom, epsilon=self._bn_eps)
        self._swish =nn.Swish()

    def _init_weights(self):
        weight_attr = paddle.ParamAttr(
            initializer=paddle.nn.initializer.KaimingUniform())
        bias_attr = paddle.ParamAttr(
            initializer=paddle.nn.initializer.Constant(value=0.0))
        return weight_attr, bias_attr

    def forward(self, inputs, drop_connect_rate=None):
        """
        :param inputs: input tensor
        :param drop_connect_rate: drop connect rate (float, between 0 and 1)
        :return: output of block
        """

        # Expansion and Depthwise Convolution
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = self._swish(self._bn0(self._expand_conv(inputs)))
        x = self._swish(self._bn1(self._depthwise_conv(x)))

        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_expand(self._swish(self._se_reduce(x_squeezed)))
            x = F.sigmoid(x_squeezed) * x

        x = self._bn2(self._project_conv(x))

        # Skip connection and drop connect
        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs  # skip connection
        return x



class EfficientNet(nn.Layer):
    """
    An EfficientNet model. Most easily loaded with the .from_name or .from_pretrained methods
    Args:
        blocks_args (list): A list of BlockArgs to construct blocks
        global_params (namedtuple): A set of GlobalParams shared between blocks
    Example:
        model = EfficientNet.from_pretrained('efficientnet-b0')
    """

    def __init__(self, blocks_args=None, global_params=None):
        # blocks_args, global_params: configs for original EfficientNet
        # d_model, max_n_chars, n_head: configs for additional parts

        super().__init__()
        assert isinstance(blocks_args, list), 'blocks_args should be a list'
        assert len(blocks_args) > 0, 'block args must be greater than 0'
        self._global_params = global_params
        self._blocks_args = blocks_args

        # Get static or dynamic convolution depending on image size
        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size) #Conv2dStaticSamePadding

        # Batch norm parameters
        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon

        # Stem
        w_attr_1, b_attr_1 = self._init_weights()
        in_channels = 3  # rgb
        out_channels = round_filters(32, self._global_params)  # number of output channels
        self._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias_attr=False,weight_attr=w_attr_1)
        self._bn0 = nn.BatchNorm2D(num_features=out_channels, momentum=bn_mom, epsilon=bn_eps)

        # Build blocks
        self._blocks =[]
        idx = 0
        block_size = 0
        current__stage = 1
        for block_args in self._blocks_args:

            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, self._global_params),
                output_filters=round_filters(block_args.output_filters, self._global_params),
                num_repeat=round_repeats(block_args.num_repeat, self._global_params)
            )

            block_size+=1
            # The first block needs to take care of stride and filter size increase.
            _mc_block=self.add_sublayer(name="_blocks." + str(idx) + ".",sublayer=MBConvBlock(block_args, self._global_params))
            self._blocks.append(_mc_block)
            #+++
            idx += 1
            if block_args.num_repeat > 1:
                block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)
            for _ in range(block_args.num_repeat - 1):
                _mc_block=self.add_sublayer(name="_blocks." + str(idx) + ".",sublayer=MBConvBlock(block_args, self._global_params))
                self._blocks.append(_mc_block)
                idx += 1
            current__stage += 1

        self._block_idx = [7, 17]  # feature maps for upsample_add
        self._swish =nn.Swish()

    def _upsample_add(self, x, y):
        _,_,H,W = y.size()
        x = F.interpolate(x, size=(H,W), mode='bilinear', align_corners=True) + y
        return self._swish(x)

    def _init_weights(self):
        weight_attr = paddle.ParamAttr(
            initializer=paddle.nn.initializer.KaimingUniform())
        bias_attr = paddle.ParamAttr(
            initializer=paddle.nn.initializer.Constant(value=0.0))
        return weight_attr, bias_attr

    def forward(self, inputs):
        # Stem
        x = self._swish(self._bn0(self._conv_stem(inputs)))

        temps = []  # temps[0]: 1/8 size, temps[1]: 1/16 size

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            if idx in self._block_idx:
                temps.append(x)

        return temps[0], temps[1], x  # [b, 48, 8, 32], [b, 136, 4, 16], [b, 384, 2, 8]

    @classmethod
    def from_name(cls, model_name, override_params=None):
        cls._check_model_name_is_valid(model_name)
        blocks_args, global_params = get_model_params(model_name, override_params)
        return cls(blocks_args, global_params)

    @classmethod
    def _check_model_name_is_valid(cls, model_name, also_need_pretrained_weights=False):
        """ Validates model name. None that pretrained weights are only available for
        the first four models (efficientnet-b{i} for i in 0,1,2,3) at the moment. """
        num_models = 4 if also_need_pretrained_weights else 8
        valid_models = ['efficientnet-b' + str(i) for i in range(num_models)]
        if model_name not in valid_models:
            raise ValueError('model_name should be one of: ' + ', '.join(valid_models))



if __name__ == '__main__':
    import random
    model = EfficientNet.from_name('efficientnet-b3')
    x = paddle.randn(shape=[2, 3, 64, 256])
    f3, f5, f7 = model(x)
    print(f3.shape, f5.shape, f7.shape) #[2, 48, 8, 32] [2, 136, 4, 16] [2, 384, 2, 8]