from Nets.model import Model
from Utils.utils import *
import paddle.nn.functional as F
import cv2
import numpy as np
import paddle
import paddle.vision.transforms as transforms
from data.data_utils import ToPILImage
from config import get_config


# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def test(model, converter, img, img_clock, img_counter, is_vert):
    model.eval()
    with paddle.no_grad():
        # ims.shape #10, 3, 64, 256
        # is_vert.shape  batch size=1
        logits = model(img)
        if is_vert:
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
        print('pred ==> {}'.format(pred))


def get_test_transforms(configs):
    transforms_test = transforms.Compose([
        ToPILImage(),
        transforms.Resize((configs.LMDB.imgH, configs.LMDB.imgW)),  # HWC
        transforms.Transpose(order=(2, 0, 1)),  # CHW
        transforms.Normalize(mean=[0, 0, 0], std=[255.0, 255.0, 255.0]),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    return transforms_test


def main(args):
    configs = get_config()
    # 读取img
    filepath = args.filepath
    img = cv2.imread(filepath, cv2.IMREAD_COLOR)
    h, w, _ = img.shape

    # test预处理
    x = get_test_transforms(configs)(img)
    x = paddle.to_tensor(x, dtype='float32')
    x = paddle.unsqueeze(x, axis=0)

    is_vert = False
    img_clock = 0
    img_counter = 0
    if h > w:
        is_vert = True
        img_clock = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)  # 旋转90
        img_counter = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)  # 旋转270
        img_clock = get_test_transforms(configs)(img_clock)
        img_counter = get_test_transforms(configs)(img_counter)

        img_clock = paddle.to_tensor(img_clock, dtype='float32')
        img_clock = paddle.unsqueeze(img_clock, axis=0)

        img_counter = paddle.to_tensor(img_counter, dtype='float32')
        img_counter = paddle.unsqueeze(img_counter, axis=0)

    model = Model(configs)
    model_state = paddle.load(configs.LMDB.testResume + '.pdparams')
    model.set_dict(model_state)
    print('[Info] Load model from {}'.format(configs.LMDB.testResume))

    with open(configs.DATA.alphabet) as f:
        alphabet = f.readline().strip()
    strconvert = strLabelConverter(alphabet)

    test(model=model, converter=strconvert, img=x, img_clock=img_clock, img_counter=img_counter, is_vert=is_vert)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('pren')
    parser.add_argument('--filepath', type=str, default=None)
    args = parser.parse_args()
    main(args)
