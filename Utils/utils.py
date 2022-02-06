import paddle
import numpy as np


class strLabelConverter(object):
    """Convert between str and label.

    Args:
        alphabet (str): set of the possible characters.
    """

    def __init__(self, alphabet, maxT=25):
        self.alphabet = alphabet
        self.maxT = maxT

        self.dict = {}

        self.dict['<pad>'] = 0  # pad
        self.dict['<eos>'] = 1  # EOS
        self.dict['<unk>'] = 2  # OOV
        for i, item in enumerate(self.alphabet):
            self.dict[item] = i + 3  # encoding from 3 for characters in alphabet

        self.chars = list(self.dict.keys())

    def encode(self, text):
        """
        Args:
            text (list of str): texts to convert.
        Returns:
            torch.IntTensor targets: [b, L]
        """

        tars = []
        for s in text:
            tar = []
            for c in s:
                if c in self.dict.keys():
                    tar.append(self.dict[c])
                else:
                    tar.append(self.dict['<unk>'])
            tars.append(tar)

        b = len(tars)  # b
        targets = self.dict['<pad>'] * np.ones(shape=[b, self.maxT])

        for i in range(b):  # 字段填充 用eos
            if len(tars[i]) >= self.maxT:
                targets[i] = tars[i][0:self.maxT]  # 如果label大于25，那么就截断到25
            else:
                targets[i][:len(tars[i])] = tars[i]
                targets[i][len(tars[i])] = self.dict['<eos>']

        return paddle.to_tensor(targets, dtype='int64')

    def decode(self, t):
        texts = [self.chars[i] for i in t]
        return ''.join(texts)


class AverageMeter(object):
    """ Meter for monitoring losses"""

    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0
        self.reset()

    def reset(self):
        """reset all values to zeros"""
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        """update avg by val and n, where val is the avg of n values"""
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt
