# Copyright 2019 Xilinx Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# PART OF THIS FILE AT ALL TIMES.

import random as prandom
import types

import cv2
import imgaug.augmenters as iaa
import numpy as np
import utils.Automold as am
from numpy import random


class Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, seg=None):
        args = [img, seg]
        for t in self.transforms:
            args = t(*args)
        return args


class Lambda(object):
    """Applies a lambda as a transform."""

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img, seg=None):
        return self.lambd(img, seg)


class ConvertFromInts(object):
    def __call__(self, image, seg=None):
        return image.astype(np.float32), seg.astype(np.float32)


class SubtractMeans(object):
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, seg=None, info=None):
        image = image.astype(np.float32)
        image -= self.mean
        image = image / 255.0
        return image.astype(np.float32), seg, info


class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, seg=None):
        h, w, c = image.shape
        resize_scale = (self.size[1], self.size[0])
        scale_h = self.size[0] / h
        scale_w = self.size[1] / w
        info = dict(
            scale_h=scale_h,
            scale_w=scale_w
        )
        image = cv2.resize(image, resize_scale)
        return image, seg, info


class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."
        self.saturation = iaa.MultiplySaturation((lower, upper))

    def __call__(self, image, seg=None):
        if random.randint(2):
            image = image.astype('uint8')
            image = self.saturation(image=image)
            image = image.astype('float32')

        return image, seg


class RandomHue(object):
    def __init__(self, delta=18):
        assert delta >= 0 and delta <= 360
        self.delta = delta
        self.hue = iaa.AddToHue(value=(-self.delta, self.delta))

    def __call__(self, image, seg=None):
        if random.randint(2):
            image = image.astype('uint8')
            image = self.hue(image=image)
            image = image.astype('float32')
        return image, seg


class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image, seg=None):
        if random.randint(2):
            swap = self.perms[random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)  # shuffle channels
            image = shuffle(image)
        return image, seg


class ConvertColor(object):
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, image, seg=None):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        return image, seg


class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."
        self.contrast = iaa.GammaContrast((lower, upper))

    # expects float image
    def __call__(self, image, seg=None):
        if random.randint(2):
            image = image.astype('uint8')
            image = self.contrast(image=image)
            image = image.astype('float32')
        return image, seg


class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, seg=None):
        if random.randint(2):
            image = image.astype('uint8')
            image = iaa.MultiplyAndAddToBrightness(from_colorspace='BGR', add=(-self.delta, self.delta))(image=image)
            image = image.astype('float32')
        return image, seg


class RandomCropKitti(object):
    def __init__(self, aspect_ratios=(1280 / 720, 1920 / 1280, 1920 / 886, 2048 / 1024)):
        self.aspect_ratios = aspect_ratios

    def __call__(self, image, seg=None):
        aspect_ratio = random.choice(self.aspect_ratios)
        crop = iaa.CropToAspectRatio(aspect_ratio).to_deterministic()
        image, seg = crop(image=image), crop(image=seg)
        return image, seg


class RandomCrop(object):
    def __init__(self, size):
        self.h = size[0]
        self.w = size[1]

    def __call__(self, image, seg=None):
        if random.randint(2):
            h, w, c = image.shape
            min_new_h = max(self.h, int(h * 0.7))
            min_new_w = max(self.w, int(w * 0.7))

            new_h = prandom.randint(min_new_h, h)
            new_w = prandom.randint(min_new_w, w)
            crop = iaa.CropToFixedSize(width=new_w, height=new_h).to_deterministic()
            image, seg = crop(image=image), crop(image=seg)
        return image, seg


class CropKitti(object):
    def crop_img(self, img):
        height, width, channels = img.shape
        top, left = int(height - 352), int((width - 1216) / 2)
        return img[:, left:left + 1216]

    def __call__(self, img, seg):
        return self.crop_img(img), self.crop_img(seg)


class RandomMirror(object):
    def __call__(self, image, seg):
        if random.randint(2):
            image = image[:, ::-1]
            seg = seg[:, ::-1]
        return image, seg


class SwapChannels(object):
    """Transforms a tensorized image by swapping the channels in the order
     specified in the swap tuple.
    Args:
        swaps (int triple): final order of channels
            eg: (2, 1, 0)
    """

    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        """
        Args:
            image (Tensor): image tensor to be transformed
        Return:
            a tensor with channels swapped according to swap
        """
        # if torch.is_tensor(image):
        #     image = image.data.cpu().numpy()
        # else:
        #     image = np.array(image)
        image = image[:, :, self.swaps]
        return image


class RoadAug(object):
    def __call__(self, image, seg):
        image = image.astype('uint8')
        if prandom.uniform(0, 1) >= 0.7:
            image = am.add_shadow(image, no_of_shadows=prandom.choice([1, 2, 3]),
                                  shadow_dimension=prandom.choice([3, 4, 5]))
        if prandom.uniform(0, 1) >= 0.7:
            radius_min = 300
            radius_max = 400
            h, w, _ = image.shape
            ratio = max(radius_max / h, radius_max / w)
            if ratio > 1:
                radius_max = int(min(radius_max / ratio - 1, min(h, w)))
                radius_min = int(max(radius_min / ratio - 1, 0))
            image = am.add_sun_flare(image, no_of_flare_circles=prandom.choice([1, 2, 3, 4]),
                                     src_radius=prandom.randint(radius_min, radius_max))
        image = image.astype('float32')
        return image, seg


class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            RandomContrast(),
            RoadAug(),
            RandomSaturation(),
            RandomHue(),
            RandomContrast()
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()
        self.additive_gaussian_noise = iaa.Sequential([
            iaa.Sometimes(0.5, iaa.AdditiveGaussianNoise(scale=(0, 15)))
        ])

    def __call__(self, image, seg):
        im = image.copy()
        im, seg = self.rand_brightness(im, seg)
        if random.randint(2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        im, seg = distort(im, seg)
        im = im.astype('uint8')
        im = self.additive_gaussian_noise(image=im)
        im = im.astype('float32')
        return im, seg


class RandomRotate(object):
    def __init__(self, degree=1.0):
        self.degree = degree

    def __call__(self, image, seg):
        random_angle = (random.random() - 0.5) * 2 * self.degree
        image = iaa.Affine(backend='cv2', rotate=random_angle, order=cv2.INTER_LINEAR, mode='reflect')(image=image)
        seg = iaa.Affine(backend='cv2', rotate=random_angle, order=cv2.INTER_NEAREST, mode='constant')(image=seg)
        return image, seg


class KittiDepthAugmentation(object):
    def __init__(self, size, mean):
        self.mean = mean
        self.size = size
        self.augment = Compose([
            ConvertFromInts(),
            CropKitti(),
            RandomCropKitti(),
            RandomCrop(self.size),
            PhotometricDistort(),
            RandomRotate(),
            RandomMirror(),
            Resize(self.size),
            SubtractMeans(self.mean)
        ])

    def __call__(self, img, seg):
        return self.augment(img, seg)


class DepthAugmentation(object):
    def __init__(self, size, mean):
        self.mean = mean
        self.size = size
        self.augment = Compose([
            ConvertFromInts(),
            PhotometricDistort(),
            RandomCrop(self.size),
            RandomRotate(),
            RandomMirror(),
            Resize(self.size),
            SubtractMeans(self.mean)
        ])

    def __call__(self, img, seg):
        return self.augment(img, seg)
