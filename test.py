from Nets.model import Model
from Utils.utils import *
from config import get_config
import numpy as np
import paddle
import paddle.nn.functional as F
from data.lmdbTestdataset import TestLoader
import os

# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
configs = get_config()


def test(model, testloader, converter):
    model.eval()
    n_correct = 0.
    n_ims = 0
    with paddle.no_grad():
        for index, (img, label, img_clock, img_counter, is_vert) in enumerate(testloader):
            # ims.shape #10, 3, 64, 256
            # is_vert.shape  batch size=1
            logits = model(img)  # [1, L, n_class]  [1, 25, 39]

            if is_vert.numpy()[0]:

                logits_clock = model(img_clock)
                logits_counter = model(img_counter)

                score = F.log_softmax(logits[0], axis=1).max(axis=1)  # [1,25,39]
                pred = F.log_softmax(logits[0], axis=1).argmax(axis=1)
                pred = list(pred.cpu().numpy())

                score_clock = F.log_softmax(logits_clock[0], axis=1).max(axis=1)
                pred_clock = F.log_softmax(logits_clock[0], axis=1).argmax(axis=1)
                pred_clock = list(pred_clock.cpu().numpy())

                score_counter = F.log_softmax(logits_counter[0], axis=1).max(axis=1)
                pred_counter = F.log_softmax(logits_counter[0], axis=1).argmax(axis=1)
                pred_counter = list(pred_counter.cpu().numpy())

                scores = np.ones(3) * -np.inf

                if 1 in pred:
                    score = score[:pred.index(1)]
                    scores[0] = score.mean()
                if 1 in pred_clock:
                    score_clock = score_clock[:pred_clock.index(1)]
                    scores[1] = score_clock.mean()
                if 1 in pred_counter:
                    score_counter = score_counter[:pred_counter.index(1)]
                    scores[2] = score_counter.mean()

                c = scores.argmax()
                if c == 0:
                    pred = pred[:pred.index(1)]
                elif c == 1:
                    pred = pred_clock[:pred_clock.index(1)]
                else:
                    pred = pred_counter[:pred_counter.index(1)]


            else:
                pred = F.log_softmax(logits[0], axis=1).argmax(axis=1)
                pred = list(pred.cpu().numpy())
                if 1 in pred:
                    pred = pred[:pred.index(1)]

            pred = converter.decode(pred)
            pred = pred.replace('<unk>', '')
            gt = label[0]
            n_correct += (pred == gt)
            n_ims += 1
            print(100 * n_correct / n_ims)
            if configs.TEST.display :
                print('{} ==> {}  {}'.format(gt, pred, 'correct' if pred == gt else 'error'))

        print('-' * 50)
        print('Acc_word = {:.3f}%'.format(100 * n_correct / n_ims))


def main():
    testloader = TestLoader(configs)
    print('[Info] Load data from {}'.format(configs.LMDB.testDir))

    model = Model(configs)
    model_state = paddle.load(configs.LMDB.testResume + '.pdparams')
    model.set_dict(model_state)
    print('[Info] Load model from {}'.format(configs.LMDB.testResume))

    with open(configs.DATA.alphabet) as f:
        alphabet = f.readline().strip()
    strconvert = strLabelConverter(alphabet)

    test(model, testloader, converter=strconvert)


if __name__ == '__main__':
    main()
