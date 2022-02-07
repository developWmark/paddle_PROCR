import random
import PIL
import cv2
import imgaug.augmenters as iaa
import numpy as np
import paddle
from PIL import Image
from paddle.vision.transforms import BaseTransform

paddle.seed(1)
np.random.seed(1)
random.seed(1)


class Augmenter(object):
    def __init__(self, p=0.3,
                 mul_bright=(0.7, 1.3),
                 rot_angle=(-15, 15), perspective=(0, 0.06),  # rotation and perspective
                 resize=(0.5, 1.0), compression=(50, 75), th=16,  # decrease image quality
                 motionblur=(3, 7), gaussblur=(0., 2.0), p_motion=0.5,  # blur
                 p_curve=0.1):

        self.aug_bright = iaa.MultiplyBrightness(mul=mul_bright)
        self.aug_rot = iaa.Rotate(rot_angle, mode='constant', cval=0, fit_output=True)
        self.aug_perspective = iaa.PerspectiveTransform(scale=perspective, fit_output=False)

        self.aug_dq = iaa.Sequential([
            iaa.Resize(resize),
            iaa.JpegCompression(compression)
        ])

        self.aug_motion = iaa.MotionBlur(k=motionblur)
        self.aug_gauss = iaa.GaussianBlur(sigma=gaussblur)
        self.p_motion = p_motion

        self.p = p
        self.th = th
        self.p_curve = p_curve

    def apply(self, img, text_len):
        h, w, _ = img.shape

        if random.random() < self.p_curve:  # curve
            img = rand_curve(img, text_len)

        if random.random() < self.p:
            img = self.aug_bright.augment_image(img)
        if random.random() < self.p:
            img = self.aug_rot.augment_image(img)
        if random.random() < self.p:
            img = self.aug_perspective.augment_image(img)
        if random.random() < self.p:  # invert
            img = np.invert(img)
        if random.random() < self.p and min([h, w]) >= self.th:  # down quality
            img = self.aug_dq.augment_image(img)
        if random.random() < self.p:  # blur
            if random.random() < self.p_motion:
                img = self.aug_motion.augment_image(img)
            else:
                img = self.aug_gauss.augment_image(img)

        return img


def curve(img, r=1., direction=0):
    # r: the degree of curve
    # direction = 0: middle up edge down; 1: middle down edge up

    h, w, c = img.shape
    background = (img[0, 0].astype(np.int32) + img[0, w - 1].astype(np.int32) + img[h - 1, 0].astype(np.int32) + img[
        h - 1, w - 1].astype(np.int32)) / 4
    background = background.astype(np.uint8)
    dst = np.tile(background, (int(h * (1 + r)), w, 1))
    shift = int(h * r / 2 - 1e-9)  # avoid h*r/2 is int

    for j in range(w):
        delta_x = 0.5 - 4 * (j - (w - 1) / 2) ** 2 / (w - 1) ** 2
        delta_x = int(delta_x * r * h)
        if direction == 0:
            delta_x = - delta_x
        transform = np.array([[1, 0, delta_x], [0, 1, 0], [0, 0, 1]])

        for i in range(h):
            src_pos = np.array([i, j, 1])
            [x, y, _] = np.dot(transform, src_pos)
            x = int(x)
            y = int(y)
            dst[x + shift][y] = img[i][j]

    return dst


def rand_curve(img, text_len, min_text_len=4):
    if text_len < min_text_len:  # only for text length >= min_text_len
        return img

    r = random.uniform(0.2, 1.2)
    dir = 0 if random.random() < 0.5 else 1
    return curve(img, r, dir)


class ToPILImage(BaseTransform):
    def __init__(self, mode=None, keys=None):
        super(ToPILImage, self).__init__(keys)
        self.mode = mode

    def _apply_image(self, pic):
        """
        Args:
            pic (Tensor|np.ndarray): Image to be converted to PIL Image.
        Returns:
            PIL: Converted image.
        """
        if not (isinstance(pic, paddle.Tensor) or isinstance(pic, np.ndarray)):
            raise TypeError('pic should be Tensor or ndarray. Got {}.'.format(
                type(pic)))

        elif isinstance(pic, paddle.Tensor):
            if pic.ndimension() not in {2, 3}:
                raise ValueError(
                    'pic should be 2/3 dimensional. Got {} dimensions.'.format(
                        pic.ndimension()))

            elif pic.ndimension() == 2:
                # if 2D image, add channel dimension (CHW)
                pic = pic.unsqueeze(0)

        elif isinstance(pic, np.ndarray):
            if pic.ndim not in {2, 3}:
                raise ValueError(
                    'pic should be 2/3 dimensional. Got {} dimensions.'.format(
                        pic.ndim))

            elif pic.ndim == 2:
                # if 2D image, add channel dimension (HWC)
                pic = np.expand_dims(pic, 2)

        npimg = pic
        if isinstance(pic, paddle.Tensor) and "float" in str(pic.numpy(
        ).dtype) and self.mode != 'F':
            pic = pic.mul(255).byte()
        if isinstance(pic, paddle.Tensor):
            npimg = np.transpose(pic.numpy(), (1, 2, 0))

        if not isinstance(npimg, np.ndarray):
            raise TypeError(
                'Input pic must be a paddle.Tensor or NumPy ndarray, ' +
                'not {}'.format(type(npimg)))

        if npimg.shape[2] == 1:
            expected_mode = None
            npimg = npimg[:, :, 0]
            if npimg.dtype == np.uint8:
                expected_mode = 'L'
            elif npimg.dtype == np.int16:
                expected_mode = 'I;16'
            elif npimg.dtype == np.int32:
                expected_mode = 'I'
            elif npimg.dtype == np.float32:
                expected_mode = 'F'
            if self.mode is not None and self.mode != expected_mode:
                raise ValueError(
                    "Incorrect mode ({}) supplied for input type {}. Should be {}"
                        .format(self.mode, np.dtype, expected_mode))
            self.mode = expected_mode

        elif npimg.shape[2] == 2:
            permitted_2_channel_modes = ['LA']
            if self.mode is not None and self.mode not in permitted_2_channel_modes:
                raise ValueError("Only modes {} are supported for 2D inputs".
                                 format(permitted_2_channel_modes))

            if self.mode is None and npimg.dtype == np.uint8:
                self.mode = 'LA'

        elif npimg.shape[2] == 4:
            permitted_4_channel_modes = ['RGBA', 'CMYK', 'RGBX']
            if self.mode is not None and self.mode not in permitted_4_channel_modes:
                raise ValueError("Only modes {} are supported for 4D inputs".
                                 format(permitted_4_channel_modes))

            if self.mode is None and npimg.dtype == np.uint8:
                self.mode = 'RGBA'
        else:
            permitted_3_channel_modes = ['RGB', 'YCbCr', 'HSV']
            if self.mode is not None and self.mode not in permitted_3_channel_modes:
                raise ValueError("Only modes {} are supported for 3D inputs".
                                 format(permitted_3_channel_modes))
            if self.mode is None and npimg.dtype == np.uint8:
                self.mode = 'RGB'

        if self.mode is None:
            raise TypeError('Input type {} is not supported'.format(npimg.dtype))

        return Image.fromarray(npimg, mode=self.mode)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    r = '/home/yrj/Dataset/SceneText/English/'
    s = 'Synth90k/1/1/16_domestically_23176.jpg'
    im = cv2.imread(r + s)

    aug = Augmenter()
    im_aug = aug.apply(im, 4)

    f, (ax1, ax2) = plt.subplots(2, 1)
    ax1.imshow(im)
    ax2.imshow(im_aug)

