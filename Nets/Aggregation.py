import paddle
import paddle.nn as nn


class GCN(nn.Layer):
    def __init__(self, d_in, n_in, d_out=None, n_out=None, dropout=0.1):
        super().__init__()

        if d_out is None:
            d_out = d_in
        if n_out is None:
            n_out = n_in
        w_attr_1, b_attr_1 = self._init_weights()
        w_attr_2, b_attr_3 = self._init_weights()
        self.conv_n = nn.Conv1D(n_in, n_out, 1, weight_attr=w_attr_1, bias_attr=b_attr_1)
        self.linear = nn.Linear(d_in, d_out, weight_attr=w_attr_2, bias_attr=w_attr_2)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.Swish()

    def _init_weights(self):
        weight_attr = paddle.ParamAttr(
            initializer=paddle.nn.initializer.KaimingUniform())
        bias_attr = paddle.ParamAttr(
            initializer=paddle.nn.initializer.Constant(value=0.0))
        return weight_attr, bias_attr

    def forward(self, x):
        '''
        :param x: [b, nin, din]
        :return: [b, nout, dout]
        '''

        x = self.conv_n(x)  # [b, nout, din]
        x = self.dropout(self.linear(x))  # [b, nout, dout]

        return self.activation(x)


class PoolAggregate(nn.Layer):
    def __init__(self, n_r, d_in, d_middle=None, d_out=None):
        super().__init__()

        if d_middle is None:
            d_middle = d_in
        if d_out is None:
            d_out = d_in

        self.d_in = d_in
        self.d_middle = d_middle
        self.d_out = d_out
        self.activation = nn.Swish()

        self.n_r = n_r
        self.aggs = self.build_aggs()
        self.pool = nn.AdaptiveAvgPool2D(1)

    def build_aggs(self):
        aggs = nn.LayerList()

        for i in range(self.n_r):
            w_attr_1, b_attr_1 = self._init_weights()
            w_attr_2, b_attr_2 = self._init_weights()
            aggs.append(nn.Sequential(
                nn.Conv2D(self.d_in, self.d_middle, kernel_size=3, stride=2, padding=1, bias_attr=False,
                          weight_attr=w_attr_1),
                nn.BatchNorm2D(self.d_middle, momentum=0.01, epsilon=0.001),
                self.activation,
                nn.Conv2D(self.d_middle, self.d_out, kernel_size=3, stride=2, padding=1, bias_attr=False,
                          weight_attr=w_attr_2),
                nn.BatchNorm2D(self.d_out, momentum=0.01, epsilon=0.001),
            ))

        return aggs

    def _init_weights(self):
        weight_attr = paddle.ParamAttr(
            initializer=paddle.nn.initializer.KaimingUniform())
        bias_attr = paddle.ParamAttr(
            initializer=paddle.nn.initializer.Constant(value=0.0))
        return weight_attr, bias_attr

    def forward(self, x):
        '''
        :param x: [b, din, h, w]
        :return: [b, n_r, dout]
        '''

        b = x.shape[0]
        out = []
        fmaps = []

        for agg in self.aggs:
            y = agg(x)  # [b, d_out, 1, 1]
            p = self.pool(y)
            fmaps.append(y)
            # out.append(p.view(b, 1, -1))
            out.append(paddle.reshape(p, shape=[b, 1, -1]))
        out = paddle.concat(out, axis=1)  # [b, n_r, d_out]

        return out


class WeightAggregate(nn.Layer):

    def __init__(self, n_r, d_in, d_middle=None, d_out=None):
        super().__init__()

        if d_middle is None:
            d_middle = d_in
        if d_out is None:
            d_out = d_in

        w_attr_1, b_attr_1 = self._init_weights()
        w_attr_2, b_attr_2 = self._init_weights()
        w_attr_3, b_attr_3 = self._init_weights()
        w_attr_4, b_attr_4 = self._init_weights()

        self.conv_n = nn.Sequential(
            nn.Conv2D(d_in, d_in, 3, 1, 1, bias_attr=False, weight_attr=w_attr_1),
            nn.BatchNorm2D(d_in, momentum=0.01, epsilon=0.001),
            nn.Swish(),
            nn.Conv2D(d_in, n_r, 1, bias_attr=False),
            nn.BatchNorm2D(n_r, momentum=0.01, epsilon=0.001, weight_attr=w_attr_2),
            nn.Sigmoid())

        self.conv_d = nn.Sequential(
            nn.Conv2D(d_in, d_middle, 3, 1, 1, bias_attr=False, weight_attr=w_attr_3),
            nn.BatchNorm2D(d_middle, momentum=0.01, epsilon=0.001),
            nn.Swish(),
            nn.Conv2D(d_middle, d_out, 1, bias_attr=False, weight_attr=w_attr_4),
            nn.BatchNorm2D(d_out, momentum=0.01, epsilon=0.001))

        self.n_r = n_r
        self.d_out = d_out

    def _init_weights(self):
        weight_attr = paddle.ParamAttr(
            initializer=paddle.nn.initializer.KaimingUniform())
        bias_attr = paddle.ParamAttr(
            initializer=paddle.nn.initializer.Constant(value=0.0))
        return weight_attr, bias_attr

    def forward(self, x):
        '''
        :param x: [b, d_in, h, w]
        :return: [b, n_r, dout]
        '''
        b = x.shape[0]

        hmaps = self.conv_n(x)  # [b, n_r, h, w]
        fmaps = self.conv_d(x)  # [b, d_out, h, w]

        # r = paddle.bmm(hmaps.view(b, self.n_r, -1), fmaps.view(b, self.d_out, -1).permute(0, 2, 1))
        r = paddle.bmm(paddle.reshape(hmaps, shape=[b, self.n_r, -1]),
                       paddle.reshape(fmaps, shape=[b, self.d_out, -1]).transpose([0, 2, 1]))

        return r
