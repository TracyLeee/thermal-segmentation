import collections

import numpy as np
import torch
import torchvision.transforms.functional as F


class RandomCrop(object):
    def __init__(self, size, rand_gen):
        """
        :param data: Union[int, tuple[int, int]]
        """
        if isinstance(size, int):
            self.size = (size, size)
        elif (
            # collections.Iterable has been deprecated since version 3.10
            # collections.abc.Iterable was introduced in version 3.3
            isinstance(size, collections.abc.Iterable)
            and len(size) == 2
            and isinstance(size[0], int)
            and isinstance(size[1], int)
        ):
            self.size = size
        else:
            raise TypeError("Invalid crop size type")

        self.rand_gen = rand_gen

    def __call__(self, data):
        """
        :param data: tuple[PIL.Image, np.array]
        """
        image, label = data
        w, h = image.size
        width, height = self.size

        if width > w or height > h:
            raise ValueError(f"Invalide crop size: {self.size}")

        row = (
            0
            if height == h
            else int(torch.randint(h - height, (1,), generator=self.rand_gen))
        )
        col = (
            0
            if width == w
            else int(torch.randint(w - width, (1,), generator=self.rand_gen))
        )

        return (
            F.crop(image, row, col, height, width),
            label[row : row + height, col : col + width],
        )


class ColorJitter(object):
    """Randomly change the brightness, contrast, saturation and hue of an image.

    brightness (None or tuple of float (min, max)): How much to jitter brightness.
        brightness_factor is chosen uniformly from [min, max].
    contrast (None or tuple of float (min, max)): How much to jitter contrast.
        contrast_factor is chosen uniformly from [min, max].
    saturation (None or tuple of float (min, max)): How much to jitter saturation.
        saturation_factor is chosen uniformly from [min, max].
    hue (None or tuple of float (min, max)): How much to jitter hue.
        hue_factor is chosen uniformly from [min, max].
        -0.5 <= min <= max <= 0.5.
    """

    def __init__(self, brightness, contrast, saturation, hue, rand_gen):
        self.param_dict = {
            "brightness": {"min": 0, "max": float("inf")},
            "contrast": {"min": 0, "max": float("inf")},
            "brightness": {"min": 0, "max": float("inf")},
            "hue": {"min": -0.5, "max": 0.5},
        }

        self.brightness = self._init_param("brightness", brightness)
        self.contrast = self._init_param("contrast", contrast)
        self.saturation = self._init_param("saturation", saturation)
        self.hue = self._init_param("hue", hue)

        self.rand_gen = rand_gen

    def _init_param(self, param, range):
        if range is None:
            return range
        elif (
            # collections.Iterable has been deprecated since version 3.10
            # collections.abc.Iterable was introduced in version 3.3
            isinstance(range, collections.abc.Iterable)
            and len(range) == 2
            and isinstance(range[0], float)
            and isinstance(range[1], float)
        ):
            min, max = range
        else:
            raise ValueError(f"Invalid {param} value!")

        if min >= max:
            raise ValueError(f"Invalid {param} value!")

        if min < self.param_dict[param]["min"] or max > self.param_dict[param]["max"]:
            raise ValueError(f"Invalid {param} value!")

        return min, max

    def _gen_random(self, range):
        min, max = range

        return float(torch.rand(1, generator=self.rand_gen)) * (max - min) + min

    def __call__(self, data):
        """
        :param data: tuple[PIL.Image, np.array]
        """
        image, label = data

        if self.brightness is not None:
            image = F.adjust_brightness(image, self._gen_random(self.brightness))

        if self.contrast is not None:
            image = F.adjust_contrast(image, self._gen_random(self.contrast))

        if self.saturation is not None:
            image = F.adjust_saturation(image, self._gen_random(self.saturation))

        if self.hue is not None:
            image = F.adjust_hue(image, self._gen_random(self.hue))

        return image, label


class RandomHorizontalFlip(object):
    def __init__(self, prob, rand_gen):
        self.prob = prob

        self.rand_gen = rand_gen

    def __call__(self, data):
        """
        :param data: tuple[PIL.Image, np.array]
        """
        image, label = data

        if float(torch.rand(1, generator=self.rand_gen)) < self.prob:
            image = F.hflip(image)
            label = np.fliplr(label).copy()

        return image, label


class ToTensor(object):
    def __init__(self):
        pass

    def __call__(self, data):
        """
        :param data: tuple[PIL.Image, np.array]
        """
        image, label = data

        return F.to_tensor(image), torch.from_numpy(label)


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        """
        :param data: tuple[torch.Tensor, torch.Tensor]
        """
        image, label = data

        return F.normalize(image, self.mean, self.std), label


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for transform in self.transforms:
            data = transform(data)

        return data
