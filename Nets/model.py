from Nets.EfficientNet import EfficientNet
from Nets.Aggregation import GCN, WeightAggregate, PoolAggregate

import paddle
import paddle.nn as nn


class Model(nn.Layer):
    def __init__(self, configs):
        super(Model, self).__init__()

        d_model = configs.MODEL.d_model
        n_r = configs.MODEL.n_r
        max_len = configs.MODEL.max_len
        n_class = configs.MODEL.n_class
        dropout = configs.MODEL.dropout

        self.cnn = EfficientNet.from_name('efficientnet-b3')

        # pooling aggregators
        # f3[b, 48, 8, 32], f5[b, 136, 4, 16], f7[b, 384, 2, 8]
        self.agg_p1 = PoolAggregate(n_r, d_in=48, d_out=d_model // 3)
        self.agg_p2 = PoolAggregate(n_r, d_in=136, d_out=d_model // 3)
        self.agg_p3 = PoolAggregate(n_r, d_in=384, d_out=d_model // 3)

        # weighted aggregators
        self.agg_w1 = WeightAggregate(n_r=n_r, d_in=48, d_middle=4 * 48, d_out=d_model // 3)
        self.agg_w2 = WeightAggregate(n_r=n_r, d_in=136, d_middle=4 * 136, d_out=d_model // 3)
        self.agg_w3 = WeightAggregate(n_r=n_r, d_in=384, d_middle=4 * 384, d_out=d_model // 3)

        # GCNs
        self.gcn_pool = GCN(d_in=d_model, n_in=n_r, d_out=d_model, n_out=max_len, dropout=dropout)
        self.gcn_weight = GCN(d_in=d_model, n_in=n_r, d_out=d_model, n_out=max_len, dropout=dropout)

        self.linear = nn.Linear(d_model, n_class)
        self.max_len = max_len
        self.d_model = d_model

    def forward(self, input):
        '''
        :param input: images [b, 3, 64, 256]
        :return logits: [b, L, n_class] probs of characters (before softmax)
        '''

        f3, f5, f7 = self.cnn(input)

        rp1 = self.agg_p1(f3)  # [b, nr, d / 3]
        rp2 = self.agg_p2(f5)  # [b, nr, d / 3]
        rp3 = self.agg_p3(f7)  # [b, nr, d / 3]
        rp = paddle.concat([rp1, rp2, rp3], axis=2)  # [b, nr, d]

        rw1 = self.agg_w1(f3)
        rw2 = self.agg_w2(f5)
        rw3 = self.agg_w3(f7)
        rw = paddle.concat([rw1, rw2, rw3], axis=2)  # [b, nr, d]

        y1 = self.gcn_pool(rp)
        y2 = self.gcn_weight(rw)

        y = 0.5 * (y1 + y2)  # [b, L, d]

        logits = self.linear(y)

        return logits
